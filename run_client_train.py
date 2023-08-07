import os
import sys

from src import loss_functions, models
from src.clients import *
from src.clients import load_data

if __name__ == "__main__":
    # moving on ares/athena to the repo directory
    if config_train.LOCAL:
        data_dir = "C:\\Users\\JanFiszer\\data\\mega_small_hgg"
        client_id = "1"
        server_node = "127.0.0.1"
        with_num_workers = False
    else:
        os.chdir("repos/FLforMRItranslation")

        data_dir = sys.argv[1]
        client_id = sys.argv[2]
        server_node = sys.argv[3]
        with_num_workers = True
    # Model
    unet = models.UNet(batch_normalization=config_train.BATCH_NORMALIZATION).to(config_train.DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)
    criterion = loss_functions.loss_for_config()

    client = client_for_config(client_id, unet, optimizer, criterion, data_dir)

    # Address
    server_address = f"{server_node}:{config_train.PORT}"

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
