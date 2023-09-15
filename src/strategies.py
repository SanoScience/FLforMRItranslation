import time

import torch
import logging
import numpy as np

from functools import reduce
from torch.utils.data import DataLoader

from src import files_operations, loss_functions, models, datasets
from configs import config_train

import flwr as fl
from flwr.server.criterion import ClientProxy
from flwr.common import Scalar, FitRes, Parameters, logger, Metrics, NDArrays, parameters_to_ndarrays, \
    ndarrays_to_parameters
from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAdam, FedAvg, FedYogi, FedProx, FedAdagrad, FedAvgM, aggregate

from typing import List, Tuple, Dict, Union, Optional, Type
from collections import OrderedDict


def create_dynamic_strategy(StrategyClass: Type[Strategy], model: models.UNet, model_dir, *args, **kwargs):
    class SavingModelStrategy(StrategyClass):
        def __init__(self):
            initial_parameters = [val.cpu().numpy() for val in model.state_dict().values()]
            super().__init__(initial_parameters=ndarrays_to_parameters(initial_parameters), *args, **kwargs)
            self.model = model
            self.aggregation_times = []

        def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            start = time.time()

            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            # printing and saving aggregation times
            aggregation_time = time.time() - start
            self.aggregation_times.append(aggregation_time)
            print(f"\n{self.__str__()} aggregation time: {aggregation_time}\n")

            # saving in intervals
            if server_round % config_train.SAVING_FREQUENCY == 1:
                save_aggregated_model(self.model, aggregated_parameters, model_dir, server_round)

            # saving in the last round
            if server_round == config_train.N_ROUNDS:
                # model
                save_aggregated_model(self.model, aggregated_parameters, model_dir, server_round)
                # aggregation times
                with open(f"{config_train.TRAINED_MODEL_SERVER_DIR}/aggregation_times.pkl", "wb") as file:
                    pickle.dump(strategy.aggregation_times, file)

            return aggregated_parameters, aggregated_metrics

    return SavingModelStrategy()


class FedMean(FedAvg):
    def __init__(self, model, model_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_dir = model_dir

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        parameters_aggregated = ndarrays_to_parameters((self._aggregate(weights)))

        # AGGREGATING METRICS
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        # SAVING MODEL
        if parameters_aggregated is not None:
            # saving in intervals
            if server_round % config_train.SAVING_FREQUENCY == 1:
                save_aggregated_model(self.model, aggregated_parameters, model_dir, server_round)

            # saving in the last round
            if server_round == config_train.N_ROUNDS:
                # model
                save_aggregated_model(self.model, aggregated_parameters, model_dir, server_round)
                # aggregation times
                with open(f"{config_train.TRAINED_MODEL_SERVER_DIR}/aggregation_times.pkl", "wb") as file:
                    pickle.dump(strategy.aggregation_times, file)

        return parameters_aggregated, metrics_aggregated

    def _aggregate(self, results: List[NDArrays]):
        n_clients = len(results)

        aggregated_results = [
            reduce(np.add, layer) / n_clients
            for layer in results
        ]
        return aggregated_results


class FedCostWAvg(FedAvg):
    def __init__(self, model: models.UNet, model_dir, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.alpha = alpha
        self.previous_loss_values = None
        self.aggregation_times = []
        self.model_dir = model_dir

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # AGGREGATING WEIGHTS
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

        loss_values = {fit_res.metrics["client_id"]: fit_res.metrics["val_loss"] for _, fit_res in results}
        sorted_loss = OrderedDict(sorted(loss_values.items()))
        sorted_loss_values = list(sorted_loss.values())
        start = time.time()

        if self.previous_loss_values is None:
            # for the first round aggregation
            parameters_aggregated = ndarrays_to_parameters(aggregate.aggregate(weights_results))
        else:
            parameters_aggregated = ndarrays_to_parameters(self._aggregate(weights_results, sorted_loss_values))

        self.previous_loss_values = sorted_loss_values

        # printing and saving aggregation times
        aggregation_time = time.time() - start
        self.aggregation_times.append(aggregation_time)
        print(f"\n{self.__str__()} aggregation time: {aggregation_time}\n")

        # AGGREGATING METRICS
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        # SAVING MODEL
        if parameters_aggregated is not None:
            # saving in intervals
            if server_round % config_train.SAVING_FREQUENCY == 1:
                save_aggregated_model(self.model, aggregated_parameters, model_dir, server_round)

            # saving in the last round
            if server_round == config_train.N_ROUNDS:
                # model
                save_aggregated_model(self.model, aggregated_parameters, model_dir, server_round)
                # aggregation times
                with open(f"{config_train.TRAINED_MODEL_SERVER_DIR}/aggregation_times.pkl", "wb") as file:
                    pickle.dump(strategy.aggregation_times, file)

        return parameters_aggregated, metrics_aggregated

    def _aggregate(self, results: List[Tuple[NDArrays, int]], loss_values: List[Scalar]) -> NDArrays:
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # in the paper k_j = c(Mj i-1) / c(Mi j)
        Kjs = [old_loss / new_loss for old_loss, new_loss in zip(self.previous_loss_values, loss_values)]
        k_j_total = sum(Kjs)  # in the paper K = sum(k_j)

        weighted_weights = []

        # TODO: maybe optimize

        for (weights, num_examples), k_j in zip(results, Kjs):
            num_examples_normalized = self.alpha * num_examples / num_examples_total  # in the paper: alpha * (s_j)/S
            loss_factor = (1 - self.alpha) * k_j / k_j_total  # in the paper: (1 - alpha) * (k_j)/K
            weighted_weights.append([layer * (num_examples_normalized + loss_factor) for layer in weights])

        # noinspection PyTypeChecker
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates)  # potential big bug was here!!
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

    def __str__(self):
        return f"FedCostWAvg(alpha={self.alpha})"

    def __repr__(self):
        return f"FedCostWAvg(alpha={self.alpha})"


class FedPIDAvg(FedCostWAvg):
    def __init__(self, model: models.UNet, model_dir, alpha=0.45, beta=0.45, gamma=0.1, **kwargs):
        if alpha + beta + gamma != 1.0:
            ValueError(f"Alpha, beta and gamma should sum up to 1.0")

        super().__init__(model, **kwargs)
        # TODO: extra saving frequency need or not?
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.previous_loss_values = []
        self.model_dir = model_dir

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

        loss_values = {fit_res.metrics["client_id"]: fit_res.metrics["val_loss"] for _, fit_res in results}
        sorted_loss = OrderedDict(sorted(loss_values.items()))
        sorted_loss_values = list(sorted_loss.values())

        start = time.time()

        if len(self.previous_loss_values) > 0:
            parameters_aggregated = ndarrays_to_parameters(self._aggregate(weights_results, sorted_loss_values))
        else:
            # for the first round aggregation
            parameters_aggregated = ndarrays_to_parameters(aggregate.aggregate(weights_results))

        # printing and saving aggregation times
        aggregation_time = time.time() - start
        self.aggregation_times.append(aggregation_time)
        print(f"\n{self.__str__()} aggregation time: {aggregation_time}\n")

        # appending the loss to the list and keeping its size < 6
        self.previous_loss_values.append(sorted_loss_values)
        if len(self.previous_loss_values) > 5:
            self.previous_loss_values.pop(0)  # not the most optimal but the list never bigger than 5

        # AGGREGATING METRICS
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        # SAVING MODEL
        if parameters_aggregated is not None:
            # saving in intervals
            if server_round % config_train.SAVING_FREQUENCY == 1:
                save_aggregated_model(self.model, aggregated_parameters, model_dir, server_round)

            # saving in the last round
            if server_round == config_train.N_ROUNDS:
                # model
                save_aggregated_model(self.model, aggregated_parameters, model_dir, server_round)
                # aggregation times
                with open(f"{config_train.TRAINED_MODEL_SERVER_DIR}/aggregation_times.pkl", "wb") as file:
                    pickle.dump(strategy.aggregation_times, file)

        return parameters_aggregated, metrics_aggregated

    def _aggregate(self, results: List[Tuple[NDArrays, int]], loss_values: List[Scalar]):
        def sum_columns(matrix):
            num_cols = len(matrix[0])
            column_sums = [sum(row[col] for row in matrix) for col in range(num_cols)]

            return column_sums

        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        Kjs = [old_loss - new_loss
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


def save_aggregated_model(net: models.UNet, aggregated_parameters, model_dir, server_round: int, ):
    aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

    params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)

    net.save(model_dir, filename=f"round{server_round}.pth")
    logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {model_dir}")


def strategy_from_string(model, strategy_name, model_dir, evaluate_fn=None):
    drd = config_train.DATA_ROOT_DIR
    lt = config_train.LOSS_TYPE.name
    lr = config_train.LEARNING_RATE
    rd = config_train.N_ROUNDS
    ec = config_train.N_EPOCHS_CLIENT
    n = config_train.NORMALIZATION.name
    d = config_train.now.date()
    h = config_train.now.hour

    model_dir = f"{drd}/trained_models/model-{client_type_name}-{lt}-lr{lr}-rd{rd}-ep{ec}-{n}-{d}-{h}h"

    files_operations.try_create_dir(config_train.TRAINED_MODEL_SERVER_DIR)  # creating directory before to don't get warnings
    copy2("./configs/config_train.py", f"{config_train.TRAINED_MODEL_SERVER_DIR}/config.py") 

    kwargs = {
        # "evaluate_metrics_aggregation_fn": weighted_average,
        # "fit_metrics_aggregation_fn": weighted_average,
        "min_fit_clients": config_train.MIN_FIT_CLIENTS,
        "min_available_clients": config_train.MIN_AVAILABLE_CLIENTS,
        "fraction_fit": config_train.FRACTION_FIT,
        "on_fit_config_fn": get_on_fit_config(),
        "evaluate_fn": evaluate_fn,
        "on_evaluate_config_fn": get_on_eval_config()
    }
    elif strategy_name == "fedbn":
        return FedAvg(**kwargs)

    if strategy_name == "fedcostw":
        return FedCostWAvg(model, model_dir, **kwargs)
    elif strategy_name == "fedpid":
        return FedPIDAvg(model, model_dir, **kwargs)

    elif strategy_name == "fedmean":
        return FedMean(model, model_dir, ** kwargs)

    elif strategy_name == "fedprox":
        strategy_class = FedProx
        kwargs["proximal_mu"] = config_train.PROXIMAL_MU
    elif strategy_name == "fedadam":
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
    elif strategy_name == "fedavg":
        strategy_class = FedAvg
    else:
        raise ValueError(f"Wrong starategy name: {strategy_name}")

    return create_dynamic_strategy(strategy_class, model, model_dir, **kwargs)

def strategy_from_config(model, evaluate_fn=None):
    kwargs = {
        # "evaluate_metrics_aggregation_fn": weighted_average,
        # "fit_metrics_aggregation_fn": weighted_average,
        "min_fit_clients": config_train.MIN_FIT_CLIENTS,
        "min_available_clients": config_train.MIN_AVAILABLE_CLIENTS,
        "fraction_fit": config_train.FRACTION_FIT,
        "on_fit_config_fn": get_on_fit_config(),
        "evaluate_fn": evaluate_fn,
        "on_evaluate_config_fn": get_on_eval_config()
    }

    if config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_COSTW:
        return FedCostWAvg(model, **kwargs)
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_PID:
        return FedPIDAvg(model, **kwargs)
    elif config_train.CLIENT_TYPE == config_train.ClientTypes.FED_BN:
        return FedAvg(**kwargs)
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_MEAN:
        return FedMean(model, ** kwargs)

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
    else:  # FedAvg
        strategy_class = FedAvg

    return create_dynamic_strategy(strategy_class, model, **kwargs)


# FUNCTIONS
# used by the strategy to during fit and evaluate
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # val_metric_names = [f"val_{metric}" for metric in config_train.METRICS]
    results = {metric_name: 0.0 for metric_name in config_train.METRICS}

    for num_examples, m in metrics:
        for metric_name in results.keys():
            results[metric_name] += num_examples * m[metric_name]

    examples = [num_examples for num_examples, _ in metrics]

    for metric_name in results.keys():
        results[metric_name] /= sum(examples)

    return results


def get_on_fit_config():
    def on_fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_PROX:
            fit_config["drop_client"] = False

        return fit_config

    return on_fit_config_fn


def get_on_eval_config():
    def on_eval_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return on_eval_config_fn


def get_evaluate_fn(model: models.UNet,
                    client_names: List[str],
                    loss_history: Optional[Dict] = None,
                    ssim_history: Optional[Dict] = None):
    """
    This function assumes server ability to access the data. It might be against FL idea/constrains.
    It is just for measurement/evaluation purposes
    """

    raise NotImplementedError("New metrics: PNSR and MSE not include here!!")

    if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_BN:
        return None

    testsets = []
    testloaders = []

    for eval_dir in config_train.EVAL_DATA_DIRS:
        testset = datasets.MRIDatasetNumpySlices([eval_dir])
        testsets.append(testset)
        testloaders.append(DataLoader(testset,
                                      batch_size=config_train.BATCH_SIZE,
                                      num_workers=config_train.NUM_WORKERS,
                                      pin_memory=True))

    def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        param_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        model.load_state_dict(state_dict)

        total_loss, total_ssim = 0.0, 0.0

        criterion = loss_functions.loss_for_config()
        # for global evaluation we will not have global parameters so the loss function mustn't be LossWithProximalTerm
        if isinstance(criterion, loss_functions.LossWithProximalTerm):
            criterion = criterion.base_loss_fn

        print("TESTING...")

        for client_name, testloader in zip(client_names, testloaders):
            loss, ssim = model.evaluate(testloader, criterion)
            # TODO: consider if server_round needed
            loss_history[client_name].append(loss)
            ssim_history[client_name].append(ssim)

            total_loss += loss
            total_ssim += ssim

        print("END OF SERVER TESTING.")

        return total_loss / len(client_names), {"ssim": total_ssim / len(client_names)}

    return evaluate


def client_names_from_eval_dirs():
    if config_train.LOCAL:
        sep = "\\"
    else:
        sep = "/"

    return [eval_dir.split(sep)[-2] for eval_dir in config_train.EVAL_DATA_DIRS]

# class SaveModelStrategy:
#     def __init__(self, model, aggregation_class: Type[Strategy], saving_frequency=1, *args, **kwargs):
#         self.model = model
#         self.saving_frequency = saving_frequency
#         self.aggregation_class = aggregation_class.__init__(*args, **kwargs)
#
#     def aggregate_fit(
#             self,
#             server_round: int,
#             results: List[Tuple[ClientProxy, FitRes]],
#             failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
#     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
#         print(results)
#         print(failures)
#
#         aggregated_parameters, aggregated_metrics = self.aggregation_class.aggregate_fit(server_round, results,
#                                                                                          failures)
#
#         if aggregated_parameters is not None:
#             if server_round % self.saving_frequency == 0:
#                 aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
#
#                 params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
#                 state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#                 self.model.load_state_dict(state_dict)
#
#                 directory = config_train.TRAINED_MODEL_SERVER_DIR
#
#                 files_operations.try_create_dir(directory)  # creating directory before to don't get warnings
#                 self.model.save(directory, filename=f"round{server_round}.pth", create_dir=False)
#
#                 # logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")
#
#         if aggregated_metrics is not None:
#             logger.log("Aggregated metrics: ", aggregated_metrics)
#             logger.log("Aggregated metrics type: ", type(aggregated_metrics))
#
#         return aggregated_parameters, aggregated_metrics
#
#     def __getattr__(self, arg):
#         return getattr(self.aggregation_class, arg)
