from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from configs import config_train
from src import models


class ClassicClient(fl.client.NumPyClient):
    def __init__(self, client_id, model: models.UNet, optimizer, criterion,
                 train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        models.train(self.model,
                     self.train_loader,
                     self.optimizer,
                     self.criterion,
                     validationloader=self.val_loader,
                     epochs=config_train.N_EPOCHS_CLIENT)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        # TODO: valset instead of test
        self.set_parameters(parameters)
        loss, ssim = models.evaluate(self.model, self.test_loader, self.criterion)
        return loss, len(self.test_loader.dataset), {"ssim": ssim}


class FedProxClient(ClassicClient):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(self, client_id, model: models.UNet, optimizer, criterion,
                 train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader, straggler_schedule=None):  # pylint: disable=too-many-arguments
        super().__init__(client_id, model, optimizer, criterion,
                         train_loader, test_loader, val_loader)

        self.straggler_schedule = straggler_schedule

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

        if self.straggler_schedule is not None:
            if self.straggler_schedule[int(config["curr_round"]) - 1]:
                num_epochs = np.random.randint(1, config_train.N_EPOCHS_CLIENT)

                if config["drop_client"]:
                    # return without doing any training.
                    # The flag in the metric will be used to tell the strategy
                    # to discard the model upon aggregation
                    return (
                        self.get_parameters({}),
                        len(self.train_loader),
                        {"is_straggler": True},
                    )

        models.train(
            self.model,
            self.train_loader,
            self.optimizer,
            self.criterion,
            epochs=num_epochs,
            prox_loss=True,
            validationloader=self.val_loader,
        )

        return self.get_parameters({}), len(self.train_loader.dataset), {"is_straggler": False}


class FedBNClient(ClassicClient):
    def set_parameters(self, parameters):
        self.model.train()

        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        param_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=False)


def client_for_config(client_id, unet, optimizer, criterion, trainloader, testloader, valloader):
    if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_PROX:
        stragglers_mat = np.transpose(
            np.random.choice([0, 1], size=config_train.N_ROUNDS,
                             p=[1 - config_train.STRAGGLERS, config_train.STRAGGLERS])
        )

        return FedProxClient(client_id, unet, optimizer, criterion, trainloader, testloader, valloader, stragglers_mat)

    elif config_train.CLIENT_TYPE == config_train.ClientTypes.FED_BN:
        return FedBNClient(client_id, unet, optimizer, criterion, trainloader, testloader, valloader)

    else:  # config_train.CLIENT_TYPE == config_train.ClientTypes.FED_AVG:
        return ClassicClient(client_id, unet, optimizer, criterion, trainloader, testloader, valloader)
