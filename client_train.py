import sys
from collections import OrderedDict

import flwr as fl
import torch
import os

from common import models, datasets, config_train
from client.utils import train, test, load_data

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")


if __name__ == "__main__":
    # Loading data
    # data_dir = sys.argv[1]
    # client_id = sys.argv[2]
    data_dir = os.path.join(config_train.DATA_DIR, "small_hgg")
    train_loader, test_loader, val_loader = load_data(data_dir)

    # Model
    unet = models.UNet().to(config_train.DEVICE)
    optimizer = config_train.OPTIMIZER(unet.parameters(), lr=config_train.LEARNING_RATE)


    class TranslationClient(fl.client.NumPyClient):
        def __init__(self, client_id):
            self.client_id = client_id
        def get_parameters(self, config):
            return [val.cpu().numpy() for val in unet.state_dict().values()]

        def set_parameters(self, parameters):
            param_dict = zip(unet.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
            unet.load_state_dict(state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(unet, train_loader, val_loader, optimizer, epochs=config_train.N_EPOCHS_CLIENT)
            return self.get_parameters(config={}), len(train_loader.dataset), {}

        def evaluate(self, parameters, config):
            # TODO: maybe input the test_dir instead of loader
            self.set_parameters(parameters)
            loss, ssim = test(unet, test_loader)
            return loss, len(test_loader.dataset), {"ssim": ssim}


    fl.client.start_numpy_client(
        server_address=config_train.CLIENT_IP_ADDRESS,
        client=TranslationClient(client_id=1)
    )