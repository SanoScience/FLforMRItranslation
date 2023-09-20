import os
import sys

from src import loss_functions, models
from src.clients import *


if __name__ == "__main__":
    # moving on ares/athena to the repo directory
    if config_train.LOCAL:
        data_dir = "C:\\Users\\JanFiszer\\data\\mega_small_hgg"
        client_id = "1"
        server_address = "127.0.0.1"
        with_num_workers = False
    else:
        os.chdir("repos/FLforMRItranslation")

        data_dir = sys.argv[1]
        client_id = sys.argv[2]
        server_address = sys.argv[3]
        with_num_workers = True
    # Model
    criterion = loss_functions.loss_for_config()
    unet = models.UNet(criterion).to(config_train.DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)
    
    if len(sys.argv) < 5:
        client = client_for_config(client_id, unet, optimizer, data_dir)
    else:
        client = client_from_string(client_id, unet, optimizer, data_dir, sys.argv[4])
    # Address
    # If port not provided it is taken from the config
    if ":" not in server_address:
        server_address = f"{server_address}:{config_train.PORT}"

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
