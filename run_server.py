import pickle
import socket
import os

from src.strategies import *

import flwr as fl

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")

# TODO: save metrics_distributed and losses_distributed


if __name__ == "__main__":
    unet = models.UNet()

    loss_history = []
    ssim_history = []

    evaluate_fn = None
    on_fit_config = None

    if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_BN:
        evaluate_fn = get_evaluate_fn(unet, loss_history, ssim_history)

    if config_train.CLIENT_TYPE == config_train.ClientTypes.FED_PROX:
        on_fit_config = get_on_fit_config()

    saving_strategy = create_dynamic_strategy(FedAvg, unet,
                                              evaluate_metrics_aggregation_fn=weighted_average,
                                              min_fit_clients=config_train.MIN_FIT_CLIENTS,
                                              min_available_clients=config_train.MIN_AVAILABLE_CLIENTS,
                                              fraction_fit=config_train.FRACTION_FIT,
                                              on_fit_config_fn=get_on_fit_config(),
                                              evaluate_fn=evaluate_fn)

    if config_train.LOCAL:
        server_address = f"0.0.0.0:{config_train.PORT}"
    else:
        server_address = f"{socket.gethostname()}:{config_train.PORT}"

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config_train.N_ROUNDS),
        strategy=saving_strategy
    )

    history = {"loss": [val for _, val in loss_history], "ssim": [val for _, val in ssim_history]}

    with open(f"{config_train.TRAINED_MODEL_SERVER_DIR}/history.pkl", "wb") as file:
        pickle.dump(history, file)
