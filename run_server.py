import socket
import os

from src.strategies import *

import flwr as fl

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")

# TODO: save metrics_distributed and losses_distributed


if __name__ == "__main__":
    unet = models.UNet()

    strategy = SaveModelFedAvg(unet,
                               evaluate_metrics_aggregation_fn=weighted_average,
                               min_fit_clients=config_train.MIN_FIT_CLIENTS,
                               min_available_clients=config_train.MIN_AVAILABLE_CLIENTS,
                               fraction_fit=config_train.FRACTION_FIT,
                               on_fit_config_fn=get_on_fit_config(),
                               evaluate_fn=get_evaluate_fn(unet))

    fedprox_strategy = FedProxWithSave(unet,
                                       evaluate_metrics_aggregation_fn=weighted_average,
                                       min_fit_clients=config_train.MIN_FIT_CLIENTS,
                                       min_available_clients=config_train.MIN_AVAILABLE_CLIENTS,
                                       fraction_fit=config_train.FRACTION_FIT,
                                       on_fit_config_fn=get_on_fit_config(),
                                       evaluate_fn=get_evaluate_fn(unet),
                                       proximal_mu=config_train.PROXIMAL_MU)

    if config_train.LOCAL:
        server_address = f"0.0.0.0:{config_train.PORT}"
    else:
        server_address = f"{socket.gethostname()}:{config_train.PORT}"

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config_train.N_ROUNDS),
        strategy=strategy
    )
