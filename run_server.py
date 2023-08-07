import pickle
import socket
import os
import sys

from src.strategies import *

import flwr as fl

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")

# TODO: save metrics_distributed and losses_distributed


if __name__ == "__main__":
    if config_train.LOCAL:
        eval_data_dir = "C:\\Users\\JanFiszer\\data\\mega_small_hgg\\test"
    else:
        eval_data_dir = sys.argv[1]

    unet = models.UNet(batch_normalization=config_train.BATCH_NORMALIZATION)

    # TODO: maybe already init as a dict instead of two lists
    loss_history = []
    ssim_history = []

    evaluate_fn = get_evaluate_fn(unet, eval_data_dir, loss_history, ssim_history)
    saving_strategy = strategy_from_config(unet, evaluate_fn)

    if config_train.LOCAL:
        server_address = f"0.0.0.0:{config_train.PORT}"
    else:
        server_address = f"{socket.gethostname()}:{config_train.PORT}"

    print("SERVER STARTING...")
    print("Strategy used: {}\n".format(saving_strategy))

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config_train.N_ROUNDS),
        strategy=saving_strategy
    )

    if len(loss_history) > 0:
        history = {"loss": [val for _, val in loss_history], "ssim": [val for _, val in ssim_history]}

        with open(f"{config_train.TRAINED_MODEL_SERVER_DIR}/history.pkl", "wb") as file:
            pickle.dump(history, file)
