import torch
import logging
import numpy as np

from functools import reduce
from flwr.server import ClientManager
from torch.utils.data import DataLoader

from src import files_operations, loss_functions, models, datasets
from configs import config_train

import flwr as fl
from flwr.server.criterion import ClientProxy
from flwr.common import Scalar, FitRes, Parameters, logger, Metrics, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAdam, FedAvg, FedYogi, FedProx, FedAdagrad, aggregate

from typing import List, Tuple, Dict, Union, Optional, Type
from collections import OrderedDict


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    ssim_values = [num_examples * m["ssim"] for num_examples, m in metrics]
    loss_values = [num_examples * m["loss"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    results = {"loss": sum(loss_values) / sum(examples), "ssim": sum(ssim_values) / sum(examples)}

    logger.log(logging.INFO, f"Maybe here to get the results {results}")

    return results


def get_on_fit_config():
    def fit_config_fn(server_round: int):
        # resolve and convert to python dict
        fit_config = {"drop_client": False, "curr_round": server_round}
        return fit_config

    return fit_config_fn


def get_evaluate_fn(model: models.UNet, loss_history=None, ssim_history=None):
    data_dir = "C:\\Users\\JanFiszer\\data\\mega_small_hgg\\test"
    if data_dir is None:
        raise NotImplementedError
    testset = datasets.MRIDatasetNumpySlices([data_dir])
    testloader = DataLoader(testset, batch_size=config_train.BATCH_SIZE, num_workers=config_train.NUM_WORKERS,
                            pin_memory=True)

    def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        param_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        model.load_state_dict(state_dict)

        criterion = loss_functions.loss_for_config()

        # for global evaluation we will not have global parameters so the loss function mustn't be LossWithProximalTerm
        if isinstance(criterion, loss_functions.LossWithProximalTerm):
            criterion = criterion.base_loss_fn

        loss, ssim = models.evaluate(model, testloader, criterion)

        loss_history.append((server_round, loss))
        ssim_history.append((server_round, ssim))

        return loss, {"ssim": ssim}

    return evaluate


def create_dynamic_strategy(StrategyClass: Type[Strategy], model: models.UNet, saving_frequency=1, *args, **kwargs):
    class SavingModelStrategy(StrategyClass):
        def __init__(self):
            initial_parameters = [val.cpu().numpy() for val in model.state_dict().values()]
            super().__init__(initial_parameters=ndarrays_to_parameters(initial_parameters), *args, **kwargs)
            self.model = model
            self.saving_frequency = saving_frequency

        def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                if server_round % self.saving_frequency == 0:
                    aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

                    params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                    self.model.load_state_dict(state_dict)

                    directory = config_train.TRAINED_MODEL_SERVER_DIR

                    files_operations.try_create_dir(directory)  # creating directory before to don't get warnings
                    self.model.save(directory, filename=f"round{server_round}.pth", create_dir=False)

                    logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")

            # if aggregated_metrics is not None:
            #     print("Aggregated metrics: ", aggregated_metrics)
            #     print("Aggregated metrics type: ", type(aggregated_metrics))

            return aggregated_parameters, aggregated_metrics

    return SavingModelStrategy()


class SaveModelStrategy:
    def __init__(self, model, aggregation_class: Type[Strategy], saving_frequency=1, *args, **kwargs):
        self.model = model
        self.saving_frequency = saving_frequency
        self.aggregation_class = aggregation_class.__init__(*args, **kwargs)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(results)
        print(failures)

        aggregated_parameters, aggregated_metrics = self.aggregation_class.aggregate_fit(server_round, results,
                                                                                         failures)

        if aggregated_parameters is not None:
            if server_round % self.saving_frequency == 0:
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict)

                directory = config_train.TRAINED_MODEL_SERVER_DIR

                files_operations.try_create_dir(directory)  # creating directory before to don't get warnings
                self.model.save(directory, filename=f"round{server_round}.pth", create_dir=False)

                # logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")

        if aggregated_metrics is not None:
            # TODO: make agratated metric not empty
            logger.log("Aggregated metrics: ", aggregated_metrics)
            logger.log("Aggregated metrics type: ", type(aggregated_metrics))

        return aggregated_parameters, aggregated_metrics

    def __getattr__(self, arg):
        return getattr(self.aggregation_class, arg)


class FedCostWAvg(FedAvg):
    def __init__(self, model: models.UNet, saving_frequency=1, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.alpha = alpha
        self.saving_frequency = saving_frequency
        self.previous_loss_values = None  # TODO: consider initializing with some values to don;t have to check if it is none

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # AGGREGATING WEIGHTS
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]


        loss_values = [fit_res.metrics for _, fit_res in results]
        print()
        print("METRICS")
        print(loss_values)
        print()
        print()

        if self.previous_loss_values is None:
            # for the first round aggregation
            parameters_aggregated = ndarrays_to_parameters(aggregate.aggregate(weights_results))
        else:
            parameters_aggregated = ndarrays_to_parameters(self._aggregate(weights_results, loss_values))

        self.previous_loss_values = loss_values

        # AGGREGATING METRICS
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        # SAVING MODEL

        if parameters_aggregated is not None:
            if server_round % self.saving_frequency == 0:
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(parameters_aggregated)

                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict)

                directory = config_train.TRAINED_MODEL_SERVER_DIR

                files_operations.try_create_dir(directory)  # creating directory before to don't get warnings
                self.model.save(directory, filename=f"round{server_round}.pth", create_dir=False)

                # logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")

        return parameters_aggregated, metrics_aggregated

    def _aggregate(self, results: List[Tuple[NDArrays, int]], loss_values: List[Scalar]) -> NDArrays:
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # in the paper k_j = c(Mj i-1) / c(Mi j)
        Kjs = [old_loss / new_loss for old_loss, new_loss in zip(self.previous_loss_values, loss_values)]
        k_j_total = sum(Kjs)  # in the paper K = sum(k_j)

        weighted_weights = []

        for (weights, num_examples), k_j in zip(results, Kjs):
            num_examples_normalized = self.alpha * num_examples / num_examples_total  # in the paper: alpha * (s_j)/S
            loss_factor = (1 - self.alpha) * k_j / k_j_total  # in the paper: (1 - alpha) * (k_j)/K
            weighted_weights.append([layer * (num_examples_normalized + loss_factor) for layer in weights])

        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime


def strategy_from_config(model, loss_history, ssim_history):
    if config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_COSTW:
        return FedCostWAvg(model,
                           evaluate_metrics_aggregation_fn=weighted_average,
                           min_fit_clients=config_train.MIN_FIT_CLIENTS,
                           min_available_clients=config_train.MIN_AVAILABLE_CLIENTS,
                           fraction_fit=config_train.FRACTION_FIT,
                           on_fit_config_fn=get_on_fit_config(),
                           evaluate_fn=get_evaluate_fn(model, loss_history, ssim_history))

    if config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_PROX:
        strategy_class = FedProx
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_ADAM:
        strategy_class = FedAdam
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_YOGI:
        strategy_class = FedYogi
    elif config_train.AGGREGATION_METHOD == config_train.AggregationMethods.FED_ADAGRAD:
        strategy_class = FedAdagrad
    else:  # FedBN and FedAvg
        strategy_class = FedAvg

    return create_dynamic_strategy(strategy_class, model,
                                   evaluate_metrics_aggregation_fn=weighted_average,
                                   min_fit_clients=config_train.MIN_FIT_CLIENTS,
                                   min_available_clients=config_train.MIN_AVAILABLE_CLIENTS,
                                   fraction_fit=config_train.FRACTION_FIT,
                                   on_fit_config_fn=get_on_fit_config(),
                                   evaluate_fn=get_evaluate_fn(model, loss_history, ssim_history)
                                   )
