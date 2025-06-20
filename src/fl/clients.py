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

from configs import config_train, enums
from src.ml import models, custom_metrics
from src.utils import files_operations as fop
from src.ml.datasets import MRIDatasetNumpySlices


class ClassicClient(fl.client.NumPyClient):
    """
        Overriding all the methods that NumPyClient requires.
    """
    def __init__(self, client_id: str, model: models.UNet, optimizer: torch.optim.Optimizer, 
                 data_dir: str, model_dir: str = config_train.TRAINED_MODEL_SERVER_DIR) -> None:
        """
            Constructor

            Parameters
            ----------
            client_id:
                Client representative, string. Usually set to the name of the dataset.
            model:
                Client model which he will update during local training
            optimizer:
                Model's optimizer, used as the loss function
            data_dir:
                Full path to the data directory, which client trains on. Requires to have inside (named exactly):
                    - train
                    - test 
                    - validation
            model_dir:
                A directory, inside which a directory based on client_id and client type is created. 
                There the client saves its model and its potential test plot, 
        """
        self.client_id = client_id
        self.model = model
        self.train_loader, self.test_loader, self.val_loader = load_data(data_dir,
                                                                         batch_size=config_train.BATCH_SIZE,
                                                                         with_num_workers=not config_train.LOCAL)

        self.optimizer = optimizer

        self.history = {f"val_{metric_name}": [] for metric_name in config_train.METRICS}

        self.client_dir = os.path.join(model_dir,
                                       f"{self.__repr__()}_client_{self.client_id}")

        fop.try_create_dir(self.client_dir)
        print(f"Client {client_id} with data from directory: {data_dir}: INITIALIZED\n")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Extract model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        """Update local model parameters from the received parameters."""
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: NDArrays, config):
        """
        Perform local model training for one federated learning round.
        
        Args:
            parameters: Current global model parameters
            config: Configuration including current round number
            
        Returns:
            tuple: (updated parameters, number of training samples, metrics dictionary)
        """
        self.set_parameters(parameters)

        current_round = config["current_round"]
        print(f"ROUND {current_round}")

        if config_train.LOCAL:
            plots_dir = None
        else:
            plots_dir = f"{self.client_dir}/rd-{current_round}_training_plots"

        history = self.model.perform_train(self.train_loader,
                                           self.optimizer,
                                           model_dir=self.client_dir,
                                           validationloader=self.val_loader,
                                           epochs=config_train.N_EPOCHS_CLIENT,
                                           plots_dir=plots_dir
                                           )

        print(f"END OF CLIENT TRAINING\n")

        val_metric_names = [f"val_{metric}" for metric in config_train.METRICS]

        # only validation metrics from the client (ensured by 'val_' suffix)
        avg_val_metric = {
            metric_name: sum([metric_value for metric_value in history[metric_name]]) / len(history[metric_name])
            for metric_name in val_metric_names}

        avg_val_metric["client_id"] = self.client_id  # client_id to keep truck of the loss properly e.g. FedCostW

        return self.get_parameters(config=config), len(self.train_loader.dataset), avg_val_metric

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data.
        """
        self.set_parameters(parameters)

        metrics = self._evaluate(current_round=config["current_round"])
        metric_without_loss = {k: v for k, v in metrics.items() if k != "val_loss"}

        return metrics["val_loss"], len(self.test_loader.dataset), metric_without_loss

    def _evaluate(self, current_round: int) -> Dict[str, float]:

        print(f"CLIENT {self.client_id} ROUND {current_round} TESTING...")

        if config_train.LOCAL:
            plots_path = None
            plot_filename = None
        else:
            plots_path = f"{self.client_dir}/test_plots"
            plot_filename = f"round-{current_round}"

        metrics = self.model.evaluate(self.test_loader,
                                      plots_path=plots_path,
                                      plot_filename=plot_filename
                                      )

        print(f"END OF CLIENT TESTING\n\n")

        # adding to the history
        for metric_name, metric_value in metrics.items():
            self.history[metric_name].append(metric_value)

        if current_round % config_train.CLIENT_SAVING_FREQ == 0:
            self.model.save(self.client_dir, f"model-rd{current_round}")

        # saving model and history if it is the last round
        if current_round == config_train.N_ROUNDS:
            self.model.save(self.client_dir)

            with open(f"{self.client_dir}/history.pkl", 'wb') as file:
                pickle.dump(self.history, file)

        return metrics

    def __repr__(self):
        return "FedAvg"


class FedProxClient(ClassicClient):  # pylint: disable=too-many-instance-attributes
    """
    FedProx algorithm client implementation that handles stragglers and proximal term optimization.
    Extends ClassicClient with proximal regularization and variable local epochs for simulating stragglers.
    
    Additional Attributes:
        straggler_schedule: Schedule determining if client is a straggler in each round
        epochs_multiplier: Factor to adjust local epochs for stragglers
    """
    NUMBER_OF_SAMPLES = 8000  
    def __init__(self, client_id: str, model: models.UNet, optimizer: torch.optim.Optimizer, 
                 data_dir: str, model_dir: str = config_train.TRAINED_MODEL_SERVER_DIR,
                 straggler_schedule: np.ndarray = None, epochs_multiplier: int = 2) -> None:

        super().__init__(client_id, model, optimizer, data_dir, model_dir)
        """
            Requires all the same parameters as ClassicClient with extra parameters for FedProx specific function. 
        """
        self.straggler_schedule = straggler_schedule
        self.epochs_multiplier = epochs_multiplier

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """FROM THE FLOWER REPOSITORY: https://github.com/adap/flower/tree/main/baselines/fedprox
          Implements distributed fit function for a given client.
        """
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
        is_straggles = False
        print(f"ROUND {current_round}")

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
                    is_straggles = True

        if config_train.LOCAL:
            plots_dir = None
        else:
            plots_dir = f"{self.client_dir}/rd-{current_round}_training_plots"

        history = self.model.perform_train(self.train_loader,
                                           self.optimizer,
                                           epochs=num_epochs,
                                           model_dir=self.client_dir,
                                           validationloader=self.val_loader,
                                           plots_dir=plots_dir,
                                           save_best_model=True
                                           )

        print(f"END OF CLIENT TRAINING\n")

        val_metric_names = [f"val_{metric}" for metric in config_train.METRICS]
        avg_val_metrics = {
            metric_name: sum([metric_value for metric_value in history[metric_name]]) / len(history[metric_name])
            for metric_name in val_metric_names}

        avg_val_metrics["is_straggler"] = is_straggles

        return self.get_parameters({}), num_samples, avg_val_metrics

    def __repr__(self):
        return "FedProx"


class FedBNClient(ClassicClient):
    """Changes only the parameters operation (set and get) skipping the normalization layers"""
    def set_parameters(self, parameters: NDArrays) -> None:
        self.model.train()

        old_state_dict = self.model.state_dict()

        # Excluding parameters of BN layers when using FedBN
        layer_names = {index: layer_name for index, layer_name in enumerate(old_state_dict.keys())
                       if "norm" not in layer_name}

        selected_parameters = [parameters[i] for i in layer_names.keys()]
        param_dict = zip(layer_names.values(), selected_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def __repr__(self):
        return f"FedBN(batch_norm={config_train.NORMALIZATION})"


class FedMRIClient(ClassicClient):
    "Changes only the parameters operation (set and get) skipping the decoder part. Only encoder in global"
    def set_parameters(self, parameters: NDArrays) -> None:
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


def client_from_config(client_id: str, unet: models.UNet, 
                      optimizer: torch.optim.Optimizer, data_dir: str) -> ClassicClient:
    """
    Factory function to create appropriate client instance based on configuration.
    
    Args:
        client_id: Unique identifier for the client
        unet: UNet model instance
        optimizer: Model optimizer
        data_dir: Directory containing training/test/validation data
        
    Returns:
        Instance of appropriate client class based on config_train.CLIENT_TYPE
    """

    if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_PROX:
        stragglers_mat = np.transpose(
            np.random.choice([0, 1], size=config_train.N_ROUNDS,
                             p=[1 - config_train.STRAGGLERS, config_train.STRAGGLERS])
        )

        return FedProxClient(client_id, unet, enums.LossFunctions.PROX, data_dir, stragglers_mat)

    elif config_train.CLIENT_TYPE == config_train.ClientTypes.FED_BN:
        return FedBNClient(client_id, unet, optimizer, data_dir)

    elif config_train.CLIENT_TYPE == config_train.ClientTypes.FED_MRI:
        return FedMRIClient(client_id, unet, optimizer, data_dir)

    else:  # config_train.CLIENT_TYPE == config_train.ClientTypes.FED_AVG:
        return ClassicClient(client_id, unet, optimizer, data_dir)


def client_from_string(client_id: str, unet: models.UNet, optimizer: torch.optim.Optimizer, 
                      data_dir: str, client_type_name: str) -> ClassicClient:
    """
    Factory function to create client instance based on string identifier.
    
    Args:
        client_id: Unique identifier for the client
        unet: UNet model instance
        optimizer: Model optimizer
        data_dir: Directory containing training/test/validation data
        client_type_name: String identifying the client type to create
        
    Returns:
        Instance of appropriate client class
        
    Raises:
        ValueError: If client_type_name is not recognized
    """

    drd = config_train.DATA_ROOT_DIR
    lt = config_train.LOSS_TYPE.name
    t = f"{config_train.TRANSLATION[0].name}{config_train.TRANSLATION[1].name}"
    lr = config_train.LEARNING_RATE
    rd = config_train.N_ROUNDS
    ec = config_train.N_EPOCHS_CLIENT
    n = config_train.NORMALIZATION.name
    d = config_train.now.date()
    h = config_train.now.hour

    model_dir = f"{drd}/trained_models/model-{client_type_name}-{lt}-{t}-lr{lr}-rd{rd}-ep{ec}-{d}"
    
    print(f"Client {client_id} has directory: {model_dir}")

    if client_type_name == "fedprox":
        stragglers_mat = np.transpose(
            np.random.choice([0, 1], size=config_train.N_ROUNDS,
                             p=[1 - config_train.STRAGGLERS, config_train.STRAGGLERS])
        )
        
        if not isinstance(unet.criterion, custom_metrics.LossWithProximalTerm):
            # raise ValueError("Wrong loss function change it in the config")
            unet.criterion = custom_metrics.LossWithProximalTerm(proximal_mu=config_train.PROXIMAL_MU, base_loss_fn=custom_metrics.DssimMseLoss())

        return FedProxClient(client_id, unet, optimizer, data_dir, model_dir, stragglers_mat)

    elif client_type_name in ["fedbn", "fedbadam"]:
        return FedBNClient(client_id, unet, optimizer, data_dir, model_dir)
    elif client_type_name in ["fedmri", "fedmix"]:
        return FedMRIClient(client_id, unet, optimizer, data_dir, model_dir)
    elif  client_type_name in  ["fedavg", "fedcostw", "fedpid", "fedavgm", "fedadam", "fedadagrad", "fedyogi", "fedmean", "fedtrimmed"]:
        return ClassicClient(client_id, unet, optimizer, data_dir, model_dir)
    
    else:
        raise ValueError(f"Given client type ('{client_type_name}') name is invalid.")


def load_data(data_dir: str, batch_size: int, with_num_workers: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, testing and validation datasets.
    
    Args:
        data_dir: Root directory containing train/test/validation subdirectories
        batch_size: Batch size for DataLoaders
        with_num_workers: Whether to use multiple workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader, val_loader)
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    val_dir = os.path.join(data_dir, "validation")

    trainset = MRIDatasetNumpySlices([train_dir], translation_direction=config_train.TRANSLATION)
    testset = MRIDatasetNumpySlices([test_dir], translation_direction=config_train.TRANSLATION)
    validationset = MRIDatasetNumpySlices([val_dir], translation_direction=config_train.TRANSLATION)

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
