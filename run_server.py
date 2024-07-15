import socket
import os
import sys

from src.strategies import *

import flwr as fl


if __name__ == "__main__":
    unet = models.UNet().to(config_train.DEVICE)

    if len(sys.argv) < 2:
        strategy = strategy_from_config(unet, None)
        print(f"Strategy taken from config.")
    else:
        strategy = strategy_from_string(unet, sys.argv[2])  
        print(f"Strategy taken from given string: {sys.argv[2]}")

    if config_train.LOCAL:
        server_address = f"0.0.0.0:8088"
    else:
        if len(sys.argv) > 1:
            port_number = sys.argv[1]
        else:
            port_number = config_train.PORT

        server_address = f"{socket.gethostname()}:{port_number}"

    print("\n\nSERVER STARTING...")
    print("Strategy utilized: {}".format(strategy))
    print("Server address: {}\n".format(server_address))

    if not config_train.LOCAL:
        with open(f"server_nodes/{sys.argv[2]}{config_train.NODE_FILENAME}", 'w') as file:  # TODO: sys.avgr..
            file.write(socket.gethostname())

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config_train.N_ROUNDS),
        strategy=strategy
    )
