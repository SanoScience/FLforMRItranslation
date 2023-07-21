from collections import OrderedDict

import flwr as fl
import torch

from common import models, datasets, config_train
from client.utils import train, test, load_data


# Loading data
data_dir = config_train.DATA_DIR
train_loader, test_loader = load_data(data_dir)

# Model
unet = models.UNet()
optimizer = config_train.OPTIMIZER(unet.parameters(), lr=config_train.LEARNING_RATE)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in unet.state_dict().values()]

    def set_parameters(self, parameters):
        param_dict = zip(unet.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        unet.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(unet, train_loader, optimizer, epochs=config_train.N_EPOCHS_CLIENT)
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, ssim = test(unet, test_loader)
        return loss, len(test_loader.dataset), {"ssim": ssim}


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)