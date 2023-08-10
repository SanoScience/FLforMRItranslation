import os
import os.path
import pickle
import random
from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from configs import config_train
from src import models
from src.datasets import MRIDatasetNumpySlices


class ClassicClient(fl.client.NumPyClient):
    def __init__(self, client_id, model: models.UNet, optimizer, criterion, data_dir):
        self.client_id = client_id
        self.model = model
        self.train_loader, self.test_loader, self.val_loader = load_data(data_dir,
                                                                         batch_size=config_train.BATCH_SIZE,
                                                                         with_num_workers=not config_train.LOCAL)

        self.optimizer = optimizer
        self.criterion = criterion

        self.history = {"loss": [], "ssim": []}
        self.client_dir = os.path.join(config_train.TRAINED_MODEL_SERVER_DIR,
                                       f"{self.__repr__()}_client_{self.client_id}")

        print(f"Client {client_id} with data from directory: {data_dir}: INITIALIZED\n")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        current_round = config["current_round"]
        print(f"ROUND {current_round}")

        history = self.model.perform_train(self.train_loader,
                                           self.optimizer,
                                           self.criterion,
                                           validationloader=self.val_loader,
                                           epochs=config_train.N_EPOCHS_CLIENT)

        print(f"END OF CLIENT TRAINING\n")

        loss_avg = sum([loss_value for loss_value in history["val_loss"]]) / len(history["val_loss"])
        ssim_avg = sum([ssim_value for ssim_value in history["val_ssim"]]) / len(history["val_ssim"])

        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": loss_avg, "ssim": ssim_avg}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        current_round = config["current_round"]

        print(f"CLIENT {self.client_id} ROUND {current_round} TESTING...")
        loss, ssim = self.model.evaluate(self.test_loader, self.criterion)

        print(f"END OF CLIENT TESTING\n\n")

        # adding to the history
        self.history["loss"].append(loss)
        self.history["ssim"].append(ssim)

        # saving model and history if it is the last round
        if current_round == config_train.N_ROUNDS:
            self.model.save(self.client_dir)

            with open(f"{self.client_dir}/history.pkl", 'wb') as file:
                pickle.dump(self.history, file)

        return loss, len(self.test_loader.dataset), {"loss": loss, "ssim": ssim}

    def __repr__(self):
        return "FedAvg"


class FedProxClient(ClassicClient):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""
    NUMBER_OF_SAMPLES = 25000

    def __init__(self, client_id, model: models.UNet, optimizer, criterion, data_dir: str,
                 straggler_schedule=None, epochs_multiplier: int = 1):  # pylint: disable=too-many-arguments
        super().__init__(client_id, model, optimizer, criterion, data_dir)

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
        num_samples = len(self.train_loader.dataset)

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

                if config["drop_client"]:
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
                                           self.criterion,
                                           epochs=num_epochs,
                                           prox_loss=True,
                                           validationloader=self.val_loader,
                                           )

        loss_avg = sum([loss_value for loss_value in history["val_loss"]]) / len(history["val_loss"])
        ssim_avg = sum([ssim_value for ssim_value in history["val_ssim"]]) / len(history["val_ssim"])

        return self.get_parameters({}), num_samples, {"loss": loss_avg, "ssim": ssim_avg, "is_straggler": False}

    def __repr__(self):
        return "FedProx"


class FedBNClient(ClassicClient):
    def set_parameters(self, parameters):
        self.model.train()

        old_state_dict = self.model.state_dict()
        keys = [k for k in old_state_dict.keys() if "bn" not in k]
        param_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def __repr__(self):
        return "FedBN"


def client_for_config(client_id, unet: models.UNet, optimizer, criterion, data_dir: str):
    if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_PROX:
        stragglers_mat = np.transpose(
            np.random.choice([0, 1], size=config_train.N_ROUNDS,
                             p=[1 - config_train.STRAGGLERS, config_train.STRAGGLERS])
        )

        return FedProxClient(client_id, unet, optimizer, criterion, data_dir, stragglers_mat)

    elif config_train.CLIENT_TYPE == config_train.ClientTypes.FED_BN:
        return FedBNClient(client_id, unet, optimizer, criterion, data_dir)

    else:  # config_train.CLIENT_TYPE == config_train.ClientTypes.FED_AVG:
        return ClassicClient(client_id, unet, optimizer, criterion, data_dir)


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
