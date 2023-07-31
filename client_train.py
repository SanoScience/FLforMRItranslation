import sys
from collections import OrderedDict

import flwr as fl
import torch
import os

from common import models, datasets, config_train
from client.utils import train, test, load_data


# TODO: client __init__ containting the model

class TranslationClient(fl.client.NumPyClient):
    def __init__(self, client_id, model: models.UNet, train_loader, test_loader, val_loader):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if config_train.LOCAL:
            # to optimize
            train(self.model, self.val_loader, optimizer, epochs=config_train.N_EPOCHS_CLIENT)
        else:
            train(self.model, self.train_loader, optimizer, validationloader=self.val_loader, epochs=config_train.N_EPOCHS_CLIENT)

        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # TODO: valset instead of test
        self.set_parameters(parameters)
        loss, ssim = test(self.model, self.test_loader)
        return loss, len(self.test_loader.dataset), {"ssim": ssim}


if __name__ == "__main__":
    # moving on ares/athena to the repo directory
    if not config_train.LOCAL:
        os.chdir("repos/FLforMRItranslation")

    # Loading data
    data_dir = sys.argv[1]
    client_id = sys.argv[2]
    server_node = sys.argv[3]
    # data_dir = os.path.join(config_train.DATA_ROOT_DIR, "small_hgg")
    trainloader, testloader, valloader = load_data(data_dir, with_num_workers=True)

    # Model
    unet = models.UNet().to(config_train.DEVICE)
    optimizer = config_train.OPTIMIZER(unet.parameters(), lr=config_train.LEARNING_RATE)

    server_address = f"{server_node}:{config_train.PORT}"

    fl.client.start_numpy_client(
        server_address=server_address,
        client=TranslationClient(client_id, unet, trainloader, testloader, valloader)
    )
