import pickle
import socket
import os
import sys
from shutil import copy2

from src.strategies import *

import flwr as fl

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")


if __name__ == "__main__":
    evaluate_fn = None
    unet = models.UNet().to(config_train.DEVICE)

    # TODO: maybe already init as a dict instead of two lists
    loss_history = []
    ssim_history = []

    if len(sys.argv) > 1:
        eval_data_dir = sys.argv[1]
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

    with open(f"{config_train.TRAINED_MODEL_SERVER_DIR}/aggregation_times.pkl", "wb") as file:
        pickle.dump(saving_strategy.aggregation_times, file)

    copy2("./configs/config_train.py", f"{config_train.TRAINED_MODEL_SERVER_DIR}/config.py")
