"""Federated Learning client training script.
Initializes and starts FL client with specified configuration."""

import sys

from src.fl.clients import *
from src.ml import custom_metrics

if __name__ == "__main__":
    # Configure paths and settings based on environment
    if config_train.LOCAL:
        # Local testing settings
        data_dir = "C:\\Users\\JanFiszer\\data\\mri\\fl-translation\\mega_small_wu_minn"
        client_id = "0"
        server_address = "127.0.0.1:8088"
        with_num_workers = False
    else:
        # Production settings from command line args
        data_dir = sys.argv[1]  # Directory containing client's data
        client_id = sys.argv[2]  # Unique client identifier
        server_address = sys.argv[3]  # Server address/port
        with_num_workers = True  # Enable parallel data loading

        # Read server node information if not running locally
        if not config_train.LOCAL:
            with open(f"server_nodes/{sys.argv[4]}{config_train.NODE_FILENAME}", 'r') as file:
                server_node = file.read()

        # Complete server address if port-only provided
        if ":" not in server_address:
            server_address = f"{server_node}:{server_address}"

    # Initialize model with loss function from config
    criterion = custom_metrics.loss_from_config()
    unet = models.UNet(criterion).to(config_train.DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)
    
    # Create client instance based on arguments or config
    if len(sys.argv) < 5:
        client = client_from_config(client_id, unet, optimizer, data_dir)
    else:
        client = client_from_string(client_id, unet, optimizer, data_dir, sys.argv[4])

    print(f"The retrieved server address is :", server_address)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
