import os
import torch.optim as optim
from torch.nn import MSELoss

from src import loss_functions
from src.datasets import *
from src.models import *


if __name__ == '__main__':
    os.chdir("repos/FLforMRItranslation")

    train_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg/train"]
    validation_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg/validation"]

    train_dataset = MRIDatasetNumpySlices(train_directories)
    validation_dataset = MRIDatasetNumpySlices(validation_directories)

    num_workers = config_train.NUM_WORKERS
    print(f"num_workers: {num_workers}")

    trainloader = DataLoader(train_dataset,
                             batch_size=config_train.BATCH_SIZE,
                             shuffle=True,
                             num_workers=config_train.NUM_WORKERS,
                             pin_memory=True)
    valloader = DataLoader(validation_dataset,
                           batch_size=config_train.BATCH_SIZE,
                           shuffle=True,
                           num_workers=config_train.NUM_WORKERS,
                           pin_memory=True)

    criterion = loss_functions.DssimMse(zoomed_ssim=True)
    unet = UNet(criterion).to(config_train.DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)

    unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config_train.N_EPOCHS_CENTRALIZED,
                           filename="model.pth",
                           model_dir=config_train.CENTRALIZED_DIR,
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
