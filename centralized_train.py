import os
import sys

import torch.optim as optim
from torch.utils.data import DataLoader

from common.datasets import *
from common.models import *
from common.config_train import *

from client.utils import train

# for ares when in the home directory
if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")

    train_directories = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/lgg/train"
    validation_directories = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/lgg/validation"
else:
    train_directories = "C:\\Users\\JanFiszer\\data\\with_val\\test"
    validation_directories = "C:\\Users\\JanFiszer\\data\\with_val\\validation"
    # ROOT_DIR_TRAIN = os.path.join(os.path.expanduser("~"), "data/HGG")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_directories = sys.argv[1]
        validation_directories = sys.argv[2]

    train_dataset = MRIDatasetNumpySlices([train_directories])
    validation_dataset = MRIDatasetNumpySlices([validation_directories])

    if len(sys.argv) > 2:
        num_workers = int(sys.argv[3])
        print(f"num_workers: {num_workers}")

        trainloader = DataLoader(train_dataset,
                                 batch_size=config_train.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
        valloader = DataLoader(validation_dataset,
                               batch_size=config_train.BATCH_SIZE,
                               shuffle=True,
                               num_workers=num_workers,
                               pin_memory=True
                               )
    else:
        trainloader = DataLoader(train_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)
        valloader = DataLoader(validation_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)

    # train_dataset = MRIDatasetNumpySlices([ROOT_DIR_TRAIN])
    # validation_dataset = MRIDatasetNumpySlices([ROOT_DIR_VAL])
    # trainloader = DataLoader(train_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)
    # valloader = DataLoader(validation_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)

    unet = UNet().to(config_train.DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)

    train(unet,
          trainloader,
          valloader,
          optimizer,
          epochs=config_train.N_EPOCHS_CLIENT,
          filename="model.pth",
          plots_dir="predictions")
