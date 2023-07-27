import sys
from collections import OrderedDict

import flwr as fl
import torch
import os

from common import models, datasets, config_train
from client.utils import train, test, load_data

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")


class TranslationClient(fl.client.NumPyClient):
    def __init__(self, client_id, evaluate=True):
        self.client_id = client_id
        self.perform_evaluate = evaluate
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in unet.state_dict().values()]

    def set_parameters(self, parameters):
        param_dict = zip(unet.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        unet.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if config_train.LOCAL:
            # to optimize
            train(unet, val_loader, optimizer, epochs=config_train.N_EPOCHS_CLIENT)
        else:
            train(unet, train_loader, optimizer, validationloader=val_loader, epochs=config_train.N_EPOCHS_CLIENT)

        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # TODO: valset instead of test
        if self.perform_evaluate:
            self.set_parameters(parameters)
            loss, ssim = test(unet, test_loader)
            return loss, len(test_loader.dataset), {"ssim": ssim}
        else:
            pass


if __name__ == "__main__":
    # Loading data
    data_dir = sys.argv[1]
    client_id = sys.argv[2]
    server_node = sys.argv[3]
    # data_dir = os.path.join(config_train.DATA_ROOT_DIR, "small_hgg")
    train_loader, test_loader, val_loader = load_data(data_dir, with_num_workers=True)

    # Model
    unet = models.UNet().to(config_train.DEVICE)
    optimizer = config_train.OPTIMIZER(unet.parameters(), lr=config_train.LEARNING_RATE)

    server_address = f"{server_node}:{config_train.PORT}"

    fl.client.start_numpy_client(
        server_address=server_address,
        client=TranslationClient(client_id=client_id, evaluate=False)
    )
