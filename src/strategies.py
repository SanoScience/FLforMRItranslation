import torch
import logging

from torch.utils.data import DataLoader

from src import files_operations, loss_functions, models, datasets
from configs import config_train

import flwr as fl
from flwr.server.criterion import ClientProxy
from flwr.common import Scalar, FitRes, Parameters, logger, Metrics, NDArrays
from flwr.server.strategy import Strategy

from typing import List, Tuple, Dict, Union, Optional
from collections import OrderedDict
from abc import ABC, abstractmethod, ABCMeta


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["ssim"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"ssim": sum(accuracies) / sum(examples)}


def get_on_fit_config():
    def fit_config_fn(server_round: int):
        # resolve and convert to python dict
        fit_config = {"drop_client": False, "curr_round": server_round}
        return fit_config

    return fit_config_fn


def get_evaluate_fn(model: models.UNet):
    data_dir = "C:\\Users\\JanFiszer\\data\\26_patients\\validation"
    if data_dir is None:
        raise NotImplementedError
    testset = datasets.MRIDatasetNumpySlices([data_dir])
    testloader = DataLoader(testset, batch_size=config_train.BATCH_SIZE, num_workers=config_train.NUM_WORKERS, pin_memory=True)

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        param_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        model.load_state_dict(state_dict)

        loss, ssim = models.evaluate(model, testloader, loss_functions.dssim_mse)

        return loss, {"ssim": ssim}

    return evaluate


class SaveModelStrategy(Strategy):
    def __init__(self, model, aggregation_class: Strategy, saving_frequency=1):
        self.model = model
        self.saving_frequency = saving_frequency
        self.aggregation_class = aggregation_class

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = self.aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            if server_round % self.saving_frequency == 0:
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict)

                directory = config_train.TRAINED_MODEL_SERVER_DIR
                files_operations.try_create_dir(directory)

                torch.save(self.model.state_dict(), f"{directory}/round{server_round}.pth")
                logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")

        return aggregated_parameters, aggregated_metrics

    def __getattr__(self, arg):
        return getattr(self.aggregation_class.__init__, arg)


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model, saving_frequency=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.saving_frequency = saving_frequency

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = general_aggregate_fit(self, server_round, results, failures)

        # aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        #
        # if aggregated_parameters is not None:
        #     aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        #
        #     params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
        #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #     self.model.load_state_dict(state_dict)
        #
        #     directory = config_train.TRAINED_MODEL_SERVER_DIR
        #     files_operations.try_create_dir(directory)
        #
        #     torch.save(self.model.state_dict(), f"{directory}/round{server_round}.pth")
        #     logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")

        return aggregated_parameters, aggregated_metrics


class FedProxWithSave(fl.server.strategy.FedProx):
    def __init__(self, model, saving_frequency=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.saving_frequency = saving_frequency

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = general_aggregate_fit(self, server_round, results, failures)

        # aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        #
        # if aggregated_parameters is not None:
        #     if server_round % self.saving_frequency == 0:
        #         aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        #
        #         params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
        #         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #         self.model.load_state_dict(state_dict)
        #
        #         directory = config_train.TRAINED_MODEL_SERVER_DIR
        #         files_operations.try_create_dir(directory)
        #
        #         torch.save(self.model.state_dict(), f"{directory}/round{server_round}.pth")
        #         logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")
        #
        return aggregated_parameters, aggregated_metrics


class FedAdamWithSave(fl.server.strategy.FedAdam):
    def __init__(self, model, saving_frequency=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.saving_frequency = saving_frequency

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = general_aggregate_fit(self, server_round, results, failures)
        # aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        #
        # if aggregated_parameters is not None:
        #     if server_round % self.saving_frequency == 0:
        #         aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        #
        #         params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
        #         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #         self.model.load_state_dict(state_dict)
        #
        #         directory = config_train.TRAINED_MODEL_SERVER_DIR
        #         files_operations.try_create_dir(directory)
        #
        #         torch.save(self.model.state_dict(), f"{directory}/round{server_round}.pth")
        #         logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")
        #
        return aggregated_parameters, aggregated_metrics


class FedAdagradWithSave(fl.server.strategy.FedAdagrad):
    def __init__(self, model, saving_frequency=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.saving_frequency = saving_frequency

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = general_aggregate_fit(self, server_round, results, failures)
        # aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        #
        # if aggregated_parameters is not None:
        #     if server_round % self.saving_frequency == 0:
        #         aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        #
        #         params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
        #         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #         self.model.load_state_dict(state_dict)
        #
        #         directory = config_train.TRAINED_MODEL_SERVER_DIR
        #         files_operations.try_create_dir(directory)
        #
        #         torch.save(self.model.state_dict(), f"{directory}/round{server_round}.pth")
        #         logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")
        #
        return aggregated_parameters, aggregated_metrics


class SavingStrategy(ABC):
    def __init__(self, model: models.UNet, saving_frequency=1):
        self.model = model
        self.saving_frequency = saving_frequency

    @abstractmethod
    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
        pass


def general_aggregate_fit(
    strategy: Strategy,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

    aggregated_parameters, aggregated_metrics = strategy.aggregate_fit(server_round, results, failures)

    if aggregated_parameters is not None:
        if server_round % strategy.saving_frequency == 0:
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            params_dict = zip(strategy.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            strategy.model.load_state_dict(state_dict)

            directory = config_train.TRAINED_MODEL_SERVER_DIR
            files_operations.try_create_dir(directory)

            torch.save(strategy.model.state_dict(), f"{directory}/round{server_round}.pth")
            logger.log(logging.INFO, f"Saved round {server_round} aggregated parameters to {directory}")

    return aggregated_parameters, aggregated_metrics

