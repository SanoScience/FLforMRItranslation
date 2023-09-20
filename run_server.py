import socket
import os
import sys

from src.strategies import *

import flwr as fl

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")


if __name__ == "__main__":
    unet = models.UNet().to(config_train.DEVICE)

    if len(sys.argv) < 2:
        strategy = strategy_from_config(unet, None)
        print(f"Strategy taken from config.")
    else:
        strategy = strategy_from_string(unet, sys.argv[2])
        print(f"Strategy taken from given string: {sys.argv[2]}")

    if config_train.LOCAL:
        server_address = f"0.0.0.0:{config_train.PORT}"
    else:
        if len(sys.argv) > 1:
            port_number = sys.argv[1]
        else:
            port_number = config_train.PORT

        server_address = f"{socket.gethostname()}:{port_number}"

    print("\n\nSERVER STARTING...")
    print("Strategy utilized: {}\n".format(strategy))

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config_train.N_ROUNDS),
        strategy=strategy
    )
