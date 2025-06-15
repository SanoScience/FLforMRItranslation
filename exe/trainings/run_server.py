"""Federated Learning server initialization script.
Sets up and starts FL server with specified strategy."""

import socket
import sys
import flwr as fl

from src.fl.strategies import *



if __name__ == "__main__":
    # Initialize model
    unet = models.UNet().to(config_train.DEVICE)

    # Choose strategy based on command line args or config
    if len(sys.argv) < 2:
        strategy = strategy_from_config(unet, None)  # Use configuration defaults
        print(f"Strategy taken from config.")
    else:
        strategy = strategy_from_string(unet, sys.argv[2])  # Use specified strategy
        print(f"Strategy taken from given string: {sys.argv[2]}")

    # Configure server address
    if config_train.LOCAL:
        server_address = f"0.0.0.0:8088"  # Local testing address
    else:
        # Use provided port or default from config
        port_number = sys.argv[1] if len(sys.argv) > 1 else config_train.PORT
        server_address = f"{socket.gethostname()}:{port_number}"

    print("\n\nSERVER STARTING...")
    print("Strategy utilized: {}".format(strategy))
    print("Server address: {}\n".format(server_address))

    # Save server node information for clients
    if not config_train.LOCAL:
        with open(f"server_nodes/{sys.argv[2]}{config_train.NODE_FILENAME}", 'w') as file:
            file.write(socket.gethostname())

    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config_train.N_ROUNDS),
        strategy=strategy
    )
