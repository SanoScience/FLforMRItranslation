"""
Federated Learning Strategy implementations for MRI translation.
Provides various aggregation methods and custom strategies for federated learning.
"""

import time
import pickle
import os
from shutil import copy2

import torch
import logging
import numpy as np

from functools import reduce

from src.ml import models
from src.utils import files_operations
from configs import config_train

import flwr as fl
from flwr.server.criterion import ClientProxy
from flwr.common import Scalar, FitRes, Parameters, logger, Metrics, NDArrays, parameters_to_ndarrays, \
    ndarrays_to_parameters, NDArray
from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAdam, FedAvg, FedYogi, FedProx, FedAdagrad, FedAvgM, aggregate

from typing import List, Tuple, Dict, Union, Optional, Type, Callable, Any
from collections import OrderedDict


def create_dynamic_strategy(StrategyClass: Type[Strategy], model: models.UNet, 
                          model_dir: str = config_train.TRAINED_MODEL_SERVER_DIR, 
                          *args, **kwargs) -> Strategy:
    """Creates a strategy class with model saving capabilities.
    
    Args:
        StrategyClass: Base strategy class to extend
        model: UNet model instance
        model_dir: Directory for saving models
        
    Returns:
        Strategy instance with model saving capabilities
    """
    class SavingModelStrategy(StrategyClass):
        def __init__(self):
            initial_parameters = [val.cpu().numpy() for val in model.state_dict().values()]
            super().__init__(initial_parameters=ndarrays_to_parameters(initial_parameters), *args, **kwargs)
            self.model = model
            self.model_dir = model_dir
            self.aggregation_times = []
            self.best_loss = float('inf')

            files_operations.try_create_dir(self.model_dir)  # creating directory before to don't get warnings
            copy2("./configs/config_train.py", f"{self.model_dir}/config.py")

        def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # start counting time
            start = time.time()

            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                        
            # printing and saving aggregation times
            aggregation_time = time.time() - start
            self.aggregation_times.append(aggregation_time)
            print(f"\n{self.__str__()} aggregation time: {aggregation_time}\n")

            # computing average loss
            loss_values = [fit_res.metrics["val_loss"] for _, fit_res in results]
            current_avg_loss = sum(loss_values)/len(loss_values)

            # saving model
            self.save_model_conditionally(aggregated_parameters, server_round, current_avg_loss=current_avg_loss)

            return aggregated_parameters, aggregated_metrics

        def save_model_conditionally(self, aggregated_parameters, server_round, save_last_round=True, save_intervals=config_train.SAVING_FREQUENCY, current_avg_loss=None):
            # saving in intervals
            if save_intervals:
                if server_round % config_train.SAVING_FREQUENCY == 1:
                    save_aggregated_model(self.model, aggregated_parameters, self.model_dir, server_round)

            # saving the best model
            if current_avg_loss:
                if current_avg_loss < self.best_loss:
                    print(f"Best model with loss {current_avg_loss:.3f}<{self.best_loss:.3f}")
                    save_aggregated_model(self.model, aggregated_parameters, self.model_dir, server_round, best_model=True)
                    self.best_loss = current_avg_loss
                else:
                    print(f"Best model with loss {current_avg_loss:.3f}>{self.best_loss:.3f}")

            # saving in the last round
            if save_last_round:
                if server_round == config_train.N_ROUNDS:
                    # model
                    save_aggregated_model(self.model, aggregated_parameters, self.model_dir, server_round)
                    # aggregation times
                    with open(f"{self.model_dir}/aggregation_times.pkl", "wb") as file:
                        pickle.dump(self.aggregation_times, file)

    return SavingModelStrategy()


class FedMean(FedAvg):
    """Implements federated averaging with equal weights for all clients."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aggregation_times = []
        self.best_loss = float('inf')

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients with equal weights."""
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        aggregated_parameters = ndarrays_to_parameters((self._aggregate(weights)))

        # AGGREGATING METRICS
        start = time.time()

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        aggregation_time = time.time() - start
        self.aggregation_times.append(aggregation_time)

        return aggregated_parameters, metrics_aggregated

    def _aggregate(self, results: List[NDArrays]):
        n_clients = len(results)
        # as in FedAvg but without num_examples
        aggregated_results = [reduce(np.add, layer) / n_clients for layer in zip(*results)]
        return aggregated_results

class FedCostWAvg(FedAvg):
    """Cost-Weighted Federated Averaging strategy.
    
    Weights client contributions based on their loss improvement and dataset size.
    
    Args:
        alpha: Weight factor between dataset size and loss improvement (default: 0.5)
    """
    
    def __init__(self, alpha: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.previous_loss_values = None
        # self.aggregation_times = []
        self.best_loss = float('inf')
        self.averaging_weights = []

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # client results might always come in a different order
        # to properly track the loss change they are sorted
        # another possibility to sort by cid from ClientProxy (ip address), maybe better?
        sorted_results = sorted(results, key=lambda x: x[1].metrics["client_id"])

        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in sorted_results]
        loss_values = [fit_res.metrics["val_loss"] for _, fit_res in sorted_results]

        if self.previous_loss_values is None:
            # the first round aggregation as normal
            aggregated_parameters = ndarrays_to_parameters(aggregate.aggregate(weights_results))
        else:
            aggregated_parameters = ndarrays_to_parameters(self._aggregate(weights_results, loss_values))

        self.previous_loss_values = loss_values

        # AGGREGATING METRICS
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in sorted_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        return aggregated_parameters, metrics_aggregated

    def _aggregate(self, results: List[Tuple[NDArrays, int]], loss_values: List[Scalar]) -> NDArrays:
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # in the paper k_j = c(Mj i-1) / c(Mi j)
        Kjs = [old_loss / new_loss for old_loss, new_loss in zip(self.previous_loss_values, loss_values)]
        k_j_total = sum(Kjs)  # in the paper K = sum(k_j)

        weighted_weights = []
        weights_num_samples = []
        weights_loss_change = []

        for (weights, num_examples), k_j in zip(results, Kjs):
            num_examples_normalized = self.alpha * num_examples / num_examples_total  # in the paper: alpha * (s_j)/S
            loss_factor = (1 - self.alpha) * k_j / k_j_total  # in the paper: (1 - alpha) * (k_j)/K
            weights_num_samples.append(num_examples_normalized)
            weights_loss_change.append(loss_factor)
            weighted_weights.append([layer * (num_examples_normalized + loss_factor) for layer in weights])

        self.averaging_weights.append({"num_examples_normalized": weights_num_samples, "loss_factor": weights_loss_change})

        # noinspection PyTypeChecker
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates)
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

    def __str__(self):
        return f"FedCostWAvg(alpha={self.alpha})"

    def __repr__(self):
        return f"FedCostWAvg(alpha={self.alpha})"


class FedPIDAvg(FedCostWAvg):
    """PID controller-based Federated Averaging strategy.
    
    A FedCostWAvg upgrade. 
    Uses proportional, integral, and derivative terms of client losses for weighting.
    
    Args:
        alpha: Weight for proportional term (default: 0.45)
        beta: Weight for integral term (default: 0.45) 
        gamma: Weight for derivative term (default: 0.1)
    """
    
    def __init__(self, alpha: float = 0.45, beta: float = 0.45, gamma: float = 0.1, **kwargs) -> None:
        if alpha + beta + gamma != 1.0:
            ValueError(f"Alpha, beta and gamma should sum up to 1.0")

        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.previous_loss_values = []

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # client results might always come in a different order
        # to properly track the loss change they are sorted
        # another possibility to sort by cid from ClientProxy (ip address), maybe better?
        sorted_results = sorted(results, key=lambda x: x[1].metrics["client_id"])

        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in sorted_results]
        loss_values = [fit_res.metrics["val_loss"] for _, fit_res in sorted_results]

        start = time.time()

        if len(self.previous_loss_values) > 0:
            aggregated_parameters = ndarrays_to_parameters(self._aggregate(weights_results, loss_values))
        else:
            # for the first round aggregation
            aggregated_parameters = ndarrays_to_parameters(aggregate.aggregate(weights_results))

        # printing and saving aggregation times
        aggregation_time = time.time() - start
        self.aggregation_times.append(aggregation_time)
        print(f"\n{self.__str__()} aggregation time: {aggregation_time}\n")

        # appending the loss to the list and keeping its size < 6
        self.previous_loss_values.append(loss_values)
        if len(self.previous_loss_values) > 5:
            self.previous_loss_values.pop(0)  # not the most optimal but the list never bigger than 5

        # AGGREGATING METRICS
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in sorted_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        return aggregated_parameters, metrics_aggregated

    def _aggregate(self, results: List[Tuple[NDArrays, int]], loss_values: List[Scalar]):
        def sum_columns(matrix):
            num_cols = len(matrix[0])
            column_sums = [sum(row[col] for row in matrix) for col in range(num_cols)]

            return column_sums

        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        Kjs = [abs(old_loss - new_loss)
               for old_loss, new_loss in
               zip(self.previous_loss_values[-1], loss_values)]  # in the paper k_j = c(Mj_i-1) - c(Mi_j)
        k_j_total = sum(Kjs)  # in the paper: K = sum(k_j)

        Mjs = sum_columns(self.previous_loss_values)  # in the paper: sum(c(M_i-l))
        m_j_total = sum(Mjs)  # in the paper: sum(m_j)
        weighted_weights = []

        for (weights, num_examples), k_j, m_j in zip(results, Kjs, Mjs):
            num_examples_normalized = self.alpha * num_examples / num_examples_total  # in the paper: alpha * (s_j)/S
            loss_difference = self.beta * k_j / k_j_total  # in the paper: beta * (k_j)/K
            loss_mean = self.gamma * m_j / m_j_total  # in the paper: gamma * m_j/I
            weighted_weights.append(
                [layer * (num_examples_normalized + loss_difference + loss_mean) for layer in weights])

        # noinspection PyTypeChecker
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates)
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

    def __str__(self):
        return f"FedPIDAvg(alpha={self.alpha}, beta{self.beta}, gamma={self.gamma})"

    def __repr__(self):
        return f"FedPIDAvg(alpha={self.alpha}, beta{self.beta}, gamma={self.gamma})"


def save_aggregated_model(model: models.UNet, 
                         aggregated_parameters: Parameters,
                         model_dir: str, 
                         server_round: int,
                         best_model: bool = False) -> None:
    """Save aggregated model parameters to disk.
    
    Args:
        model: UNet model instance
        aggregated_parameters: Parameters from federated aggregation
        model_dir: Directory to save model
        server_round: Current training round
        best_model: Whether to save as best model
    """
    aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

    params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # saving the model with an appropriate name
    model_name = "best.pth" if best_model else f"round{server_round}.pth"
    torch.save(state_dict, os.path.join(model_dir, model_name))

    # it could have been done using
    # model.load_state_dict(state_dict)
    # model.save(model_dir, filename=f"round{server_round}.pth")

    logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {model_dir}")


def strategy_from_string(model: models.UNet,
                        strategy_name: str,
                        evaluate_fn: Optional[Callable] = None) -> Strategy:
    """Create strategy instance from string identifier.
    
    Args:
        model: UNet model instance
        strategy_name: Name of strategy to create
        evaluate_fn: Optional evaluation function
        
    Returns:
        Configured strategy instance
        
    Raises:
        ValueError: If strategy_name is invalid
    """
    # the directory includes the strategy name
    # so when it is initialized by the string it is created here
    # by default it takes the name TRAINED_MODEL_SERVER_DIR
    drd = config_train.DATA_ROOT_DIR
    lt = config_train.LOSS_TYPE.name
    t = f"{config_train.TRANSLATION[0].name}{config_train.TRANSLATION[1].name}"
    lr = config_train.LEARNING_RATE
    rd = config_train.N_ROUNDS
    ec = config_train.N_EPOCHS_CLIENT
    d = config_train.now.date()

    model_dir = f"{drd}/trained_models/model-{strategy_name}-{lt}-{t}-lr{lr}-rd{rd}-ep{ec}-{d}"

    kwargs = {
        "min_fit_clients": config_train.MIN_FIT_CLIENTS,
        "min_available_clients": config_train.MIN_AVAILABLE_CLIENTS,
        "fraction_fit": config_train.FRACTION_FIT,
        "on_fit_config_fn": get_on_fit_config(),
        "evaluate_fn": evaluate_fn,
        "on_evaluate_config_fn": get_on_eval_config()
    }
    if strategy_name == "fedcostw":
        strategy_class = FedCostWAvg
        kwargs["alpha"] = config_train.ALPHA
    elif strategy_name == "fedpid":
        strategy_class = FedPIDAvg
    elif strategy_name in ["fedmean", "fedmri"]:
        strategy_class = FedMean
    elif strategy_name == "fedprox":
        strategy_class = FedProx
        kwargs["proximal_mu"] = config_train.PROXIMAL_MU
    elif strategy_name in ["fedadam", "fedmix", "fedbadam"]:
        strategy_class = FedAdam
        kwargs["tau"] = config_train.TAU
    elif strategy_name == "fedyogi":
        strategy_class = FedYogi
        kwargs["tau"] = config_train.TAU
    elif strategy_name == "fedadagrad":
        strategy_class = FedAdagrad
        kwargs["tau"] = config_train.TAU
    elif strategy_name == "fedavgm":
        strategy_class = FedAvgM
        kwargs["server_momentum"] = config_train.MOMENTUM
    elif strategy_name in ["fedavg", "fedbn"]:
        strategy_class = FedAvg
    else:
        raise ValueError(f"Wrong starategy name: {strategy_name}")

    return create_dynamic_strategy(strategy_class, model, model_dir, **kwargs)


def strategy_from_config(model, evaluate_fn=None):
    kwargs = {
        "min_fit_clients": config_train.MIN_FIT_CLIENTS,
        "min_available_clients": config_train.MIN_AVAILABLE_CLIENTS,
        "fraction_fit": config_train.FRACTION_FIT,
        "on_fit_config_fn": get_on_fit_config(),
        "evaluate_fn": evaluate_fn,
        "on_evaluate_config_fn": get_on_eval_config()
    }

    if config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_COSTW:
        strategy_class = FedCostWAvg
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_PID:
        strategy_class = FedPIDAvg
    # elif config_train.CLIENT_TYPE == config_train.ClientTypes.FED_BN:
    #     return FedAvg(**kwargs)
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_MEAN:
        strategy_class = FedMean

    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_PROX:
        strategy_class = FedProx
        kwargs["proximal_mu"] = config_train.PROXIMAL_MU
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_ADAM:
        strategy_class = FedAdam
        kwargs["tau"] = config_train.TAU
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_YOGI:
        strategy_class = FedYogi
        kwargs["tau"] = config_train.TAU
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_ADAGRAD:
        strategy_class = FedAdagrad
        kwargs["tau"] = config_train.TAU
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_AVGM:
        strategy_class = FedAvgM
        kwargs["server_momentum"] = config_train.MOMENTUM
    else:  # FedAvg or FedBN
        strategy_class = FedAvg

    return create_dynamic_strategy(strategy_class, model, **kwargs)


# FUNCTIONS
# used by the strategy to during fit and evaluate
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate weighted average of client metrics.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
        
    Returns:
        Weighted average metrics
    """
    # val_metric_names = [f"val_{metric}" for metric in config_train.METRICS]
    results = {f"val_metric_name": 0.0 for metric_name in config_train.METRICS}

    for num_examples, m in metrics:
        for metric_name in results.keys():
            results[metric_name] += num_examples * m[metric_name]

    examples = [num_examples for num_examples, _ in metrics]

    for metric_name in results.keys():
        results[metric_name] /= sum(examples)

    return results


def get_on_fit_config() -> Callable[[int], Dict[str, Any]]:
    """Returns function that generates client fit configurations."""
    def on_fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_PROX:
            fit_config["drop_client"] = False

        return fit_config

    return on_fit_config_fn


def get_on_eval_config() -> Callable[[int], Dict[str, Any]]:
    """Returns function that generates client evaluation configurations."""
    def on_eval_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return on_eval_config_fn
