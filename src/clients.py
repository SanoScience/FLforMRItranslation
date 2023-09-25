import os
import os.path
import pickle
import random
from collections import OrderedDict
from typing import Dict, Tuple, List

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from configs import config_train
from src import models, loss_functions, files_operations as fop
from src.datasets import MRIDatasetNumpySlices


class ClassicClient(fl.client.NumPyClient):
    def __init__(self, client_id, model: models.UNet, optimizer, data_dir, model_dir=config_train.TRAINED_MODEL_SERVER_DIR):
        self.client_id = client_id
        self.model = model
        self.train_loader, self.test_loader, self.val_loader = load_data(data_dir,
                                                                         batch_size=config_train.BATCH_SIZE,
                                                                         with_num_workers=not config_train.LOCAL)

        self.optimizer = optimizer

        self.history = {metric_name: [] for metric_name in config_train.METRICS}

        self.client_dir = os.path.join(model_dir,
                                       f"{self.__repr__()}_client_{self.client_id}")

        fop.try_create_dir(self.client_dir)
        print(f"Client {client_id} with data from directory: {data_dir}: INITIALIZED\n")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: NDArrays, config):
        self.set_parameters(parameters)

        current_round = config["current_round"]
        print(f"ROUND {current_round}")

        history = self.model.perform_train(self.train_loader,
                                           self.optimizer,
                                           model_dir=self.client_dir,
                                           validationloader=self.val_loader,
                                           epochs=config_train.N_EPOCHS_CLIENT,
                                           plots_dir=f"{self.client_dir}/rd-{current_round}_training_plots"
                                           )

        print(f"END OF CLIENT TRAINING\n")

        val_metric_names = [f"val_{metric}" for metric in config_train.METRICS]
        avg_val_metric = {
            metric_name: sum([metric_value for metric_value in history[metric_name]]) / len(history[metric_name])
            for metric_name in val_metric_names}

        avg_val_metric["client_id"] = self.client_id

        return self.get_parameters(config=config), len(self.train_loader.dataset), avg_val_metric

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        metrics = self._evaluate(current_round=config["current_round"])
        metric_without_loss = {k: v for k, v in metrics.items() if k != "loss"}

        return metrics["loss"], len(self.test_loader.dataset), metric_without_loss

    def _evaluate(self, current_round: int):

        print(f"CLIENT {self.client_id} ROUND {current_round} TESTING...")
        metrics = self.model.evaluate(self.test_loader,
                                      plots_path=f"{self.client_dir}/test_plots",
                                      plot_filename=f"round-{current_round}"
                                      )

        print(f"END OF CLIENT TESTING\n\n")

        # adding to the history
        for metric_name, metric_value in metrics.items():
            self.history[metric_name].append(metric_value)

        # saving model and history if it is the last round
        if current_round == config_train.N_ROUNDS:
            self.model.save(self.client_dir)

            with open(f"{self.client_dir}/history.pkl", 'wb') as file:
                pickle.dump(self.history, file)

        return metrics

    def __repr__(self):
        return "FedAvg"


class FedProxClient(ClassicClient):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""
    NUMBER_OF_SAMPLES = 1000 

    def __init__(self, client_id, model: models.UNet, optimizer, data_dir: str, model_dir=config_train.TRAINED_MODEL_SERVER_DIR,
                 straggler_schedule=None, epochs_multiplier: int = 1):  # pylint: disable=too-many-arguments
        super().__init__(client_id, model, optimizer, data_dir, model_dir)

        self.straggler_schedule = straggler_schedule
        self.epochs_multiplier = epochs_multiplier


    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        # At each round check if the client is a straggler,
        # if so, train less epochs (to simulate partial work)
        # if the client is told to be dropped (e.g. because not using
        # FedProx in the server), the fit method returns without doing
        # training.
        # This method always returns via the metrics (last argument being
        # returned) whether the client is a straggler or not. This info
        # is used by strategies other than FedProx to discard the update.
        num_epochs = config_train.N_EPOCHS_CLIENT

        current_round = config["current_round"]
        print(f"ROUND {current_round}")

        num_samples = len(self.train_loader)

        # TODO: maybe not a straggler but always like this?
        if self.straggler_schedule is not None:
            if self.straggler_schedule[int(config["current_round"]) - 1]:
                if config_train.LOCAL:
                    num_epochs = random.randint(1, 3)
                else:
                    from math import ceil
                    # inversely proportional to the number of training samples
                    # results in iterating for about the same time with different dataset size
                    num_epochs = ceil(self.epochs_multiplier * (self.NUMBER_OF_SAMPLES / num_samples))
                    # num_epochs = np.random.randint(1, config_train.N_EPOCHS_CLIENT)

                # return without doing any training.
                # The flag in the metric will be used to tell the strategy
                # to discard the model upon aggregation
                return (
                    self.get_parameters({}),
                    len(self.train_loader),
                    {"is_straggler": True},
                )

        history = self.model.perform_train(self.train_loader,
                                           self.optimizer,
                                           epochs=num_epochs,
                                           model_dir=self.client_dir,
                                           validationloader=self.val_loader,
                                           plots_dir=f"{self.client_dir}/rd-{current_round}_training_plots"
                                           )

        print(f"END OF CLIENT TRAINING\n")

        val_metric_names = [f"val_{metric}" for metric in config_train.METRICS]
        avg_val_metrics = {
            metric_name: sum([metric_value for metric_value in history[metric_name]]) / len(history[metric_name])
            for metric_name in val_metric_names}

        avg_val_metrics["is_straggler"] = False

        return self.get_parameters({}), num_samples, avg_val_metrics

    def __repr__(self):
        return "FedProx"


class FedBNClient(ClassicClient):
    def get_parameters(self, config) -> NDArrays:
        # Excluding parameters of BN layers when using FedBN
        return [val.cpu().numpy() for name, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        self.model.train()

        old_state_dict = self.model.state_dict()

        layer_names = {index: layer_name for index, layer_name in enumerate(old_state_dict.keys())
                       if "norm" not in layer_name}

        selected_parameters = [parameters[i] for i in layer_names.keys()]
        param_dict = zip(layer_names.values(), selected_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def __repr__(self):
        return f"FedBN(batch_norm={config_train.NORMALIZATION})"


class FedMRIClient(ClassicClient):
    def set_parameters(self, parameters: NDArrays):
        self.model.train()

        old_state_dict = self.model.state_dict()

        layer_names = {index: layer_name for index, layer_name in enumerate(old_state_dict.keys())
                       if "down" in layer_name or "inc" in layer_name}

        selected_parameters = [parameters[i] for i in layer_names.keys()]
        param_dict = zip(layer_names.values(), selected_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def __repr__(self):
        return f"FedMRI()"


def client_from_config(client_id, unet: models.UNet, optimizer, data_dir: str):
    if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_PROX:
        stragglers_mat = np.transpose(
            np.random.choice([0, 1], size=config_train.N_ROUNDS,
                             p=[1 - config_train.STRAGGLERS, config_train.STRAGGLERS])
        )

        return FedProxClient(client_id, unet, optimizer, data_dir, stragglers_mat)

    elif config_train.CLIENT_TYPE == config_train.ClientTypes.FED_BN:
        return FedBNClient(client_id, unet, optimizer, data_dir)

    elif config_train.CLIENT_TYPE == config_train.ClientTypes.FED_MRI:
        return FedMRIClient(client_id, unet, optimizer, data_dir)

    else:  # config_train.CLIENT_TYPE == config_train.ClientTypes.FED_AVG:
        return ClassicClient(client_id, unet, optimizer, data_dir)


def client_from_string(client_id, unet: models.UNet, optimizer, data_dir: str, client_type_name):
    drd = config_train.DATA_ROOT_DIR
    lt = config_train.LOSS_TYPE.name
    lr = config_train.LEARNING_RATE
    rd = config_train.N_ROUNDS
    ec = config_train.N_EPOCHS_CLIENT
    n = config_train.NORMALIZATION.name
    d = config_train.now.date()
    h = config_train.now.hour

    model_dir = f"{drd}/trained_models/model-{client_type_name}-{lt}-lr{lr}-rd{rd}-ep{ec}-{n}-{d}-{h}h"
    
    if client_type_name == "fedprox":
        stragglers_mat = np.transpose(
            np.random.choice([0, 1], size=config_train.N_ROUNDS,
                             p=[1 - config_train.STRAGGLERS, config_train.STRAGGLERS])
        )
        
        if not isinstance(unet.criterion, loss_functions.LossWithProximalTerm):
            raise ValueError("Wrong loss function change it in the config")

        return FedProxClient(client_id, unet, optimizer, data_dir, model_dir, stragglers_mat)

    elif client_type_name == "fedbn":
        return FedBNClient(client_id, unet, optimizer, data_dir, model_dir)
    elif client_type_name == "fedmri":
        return FedMRIClient(client_id, unet, optimizer, data_dir, model_dir)
    elif  client_type_name in  ["fedavg", "fedcostw", "fedpid", "fedavgm", "fedadam", "fedadagrad", "fedyogi", "fedmean", "fedtrimmed"]:
        return ClassicClient(client_id, unet, optimizer, data_dir, model_dir)
    
    else:
        raise ValueError(f"Given client type ('{client_type_name}') name is invalid.")


def load_data(data_dir, batch_size, with_num_workers=True):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    val_dir = os.path.join(data_dir, "validation")

    trainset = MRIDatasetNumpySlices([train_dir])
    testset = MRIDatasetNumpySlices([test_dir])
    validationset = MRIDatasetNumpySlices([val_dir])

    if with_num_workers:
        # TODO: consider persistent_workers=True
        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=config_train.NUM_WORKERS,
                                  pin_memory=True, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=config_train.NUM_WORKERS,
                                 pin_memory=True, shuffle=True)
        val_loader = DataLoader(validationset, batch_size=batch_size, num_workers=config_train.NUM_WORKERS,
                                pin_memory=True, shuffle=True)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader
