import os
import torch.optim as optim
from torch.nn import MSELoss

from src import loss_functions
from src.datasets import *
from src.models import *


if __name__ == '__main__':
    train_directories = ["C:\\Users\\JanFiszer\\data\\hgg_transformed\\validation"]
    validation_directories = ["C:\\Users\\JanFiszer\\data\\hgg_transformed\\validation"]

    train_dataset = MRIDatasetNumpySlices(train_directories)
    validation_dataset = MRIDatasetNumpySlices(validation_directories)

    num_workers = config_train.NUM_WORKERS
    print(f"num_workers: {num_workers}")
    if config_train.LOCAL:
        trainloader = DataLoader(train_dataset,
                                 batch_size=config_train.BATCH_SIZE)
        valloader = DataLoader(validation_dataset,
                               batch_size=config_train.BATCH_SIZE)
    else:
        trainloader = DataLoader(train_dataset,
                                 batch_size=config_train.BATCH_SIZE,
                                 num_workers=config_train.NUM_WORKERS,
                                 pin_memory=True)
        valloader = DataLoader(train_dataset,
                               batch_size=config_train.BATCH_SIZE,
                               num_workers=config_train.NUM_WORKERS,
                               pin_memory=True)

    criterion = loss_functions.DssimMse()
    old_unet = UNet(criterion).to(config_train.DEVICE)
    optimizer = optim.Adam(old_unet.parameters(), lr=config_train.LEARNING_RATE)

    old_unet.perform_train(trainloader, optimizer,
                           epochs=config_train.N_EPOCHS_CENTRALIZED,
                           validationloader=valloader,
                           filename="model.pth",
                           history_filename="history.pkl",
                           plots_dir="predictions")

    # unet = UNet().to(config_train.DEVICE)
    # optimizer = optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)
    # criterion = loss_functions.DssimMse()
    # model_filename = f"model.pth"

    # unet.perform_train(trainloader,
    #                    optimizer,
    #                    criterion,
    #                    validationloader=valloader,
    #                    epochs=config_train.N_EPOCHS_CENTRALIZED,
    #                    model_dir=config_train.CENTRALIZED_DIR,
    #                    filename=model_filename,
    #                    plots_dir="predictions",
    #                    history_filename="history.pkl")
