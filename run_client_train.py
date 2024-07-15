import os
import sys
import socket

from src import loss_functions, models
from src.clients import *


if __name__ == "__main__":
    # moving on ares/athena to the repo directory
    if config_train.LOCAL:
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\mega_small_wu_minn"
        client_id = "1"
        server_address = "127.0.0.1:8088"
        with_num_workers = False
    else:
        data_dir = sys.argv[1]
        client_id = sys.argv[2]
        server_address = sys.argv[3]
        with_num_workers = True

        if not config_train.LOCAL:
            with open(f"server_nodes/{sys.argv[4]}{config_train.NODE_FILENAME}", 'r') as file:   # TODO: sys.argv
                server_node = file.read()

        if ":" not in server_address:
            server_address = f"{server_node}:{server_address}"
    # Model
    criterion = loss_functions.loss_from_config()
    unet = models.UNet(criterion).to(config_train.DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)
    
    if len(sys.argv) < 5:
        client = client_from_config(client_id, unet, optimizer, data_dir)
    else:
        client = client_from_string(client_id, unet, optimizer, data_dir, sys.argv[4])
    # Address
    # If port not provided it is taken from the config

    print(f"The retrieved server address is :", server_address)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
