import os
import sys

from src import loss_functions, models
from src.clients import *
from src.datasets import load_data

if __name__ == "__main__":
    # moving on ares/athena to the repo directory
    if not config_train.LOCAL:
        os.chdir("repos/FLforMRItranslation")

        # Loading data
        data_dir = sys.argv[1]
        client_id = sys.argv[2]
        server_node = sys.argv[3]
        with_num_workers = True
    else:
        data_dir = "C:\\Users\\JanFiszer\\data\\small_hgg"
        client_id = "0"
        server_node = "127.0.0.1"
        with_num_workers = False
    # data_dir = os.path.join(config_train.DATA_ROOT_DIR, "small_hgg")
    trainloader, testloader, valloader = load_data(data_dir, config_train.BATCH_SIZE, with_num_workers=with_num_workers)

    # Model
    unet = models.UNet().to(config_train.DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)
    criterion = loss_functions.LossWithProximalTerm(config_train.PROXIMAL_MU, loss_functions.dssim_mse)
    # Address
    server_address = f"{server_node}:{config_train.PORT}"
    stragglers_mat = np.transpose(
        np.random.choice([0, 1], size=config_train.N_ROUNDS, p=[1 - config_train.STRAGGLERS, config_train.STRAGGLERS])
    )
    fl.client.start_numpy_client(
        server_address=server_address,
        client=FedProxClient(client_id, unet, optimizer, criterion, trainloader, testloader, valloader, stragglers_mat)
    )
