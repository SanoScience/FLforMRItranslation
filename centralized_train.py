import os
import sys

import torch.optim as optim
from torch.utils.data import DataLoader

from common.datasets import *
from common.models import *
from common.config_train import *

from client.utils import train

# for ares when in the home directory
os.chdir("repos/FLforMRItranslation")

train_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg_small/train"]
validation_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg_small/validation"]


if __name__ == '__main__':
    num_workers = int(sys.argv[1])

    # ROOT_DIR_TRAIN = os.path.join(os.path.expanduser("~"), "data/raw_MRI/sample")
    # ROOT_DIR_VAL = os.path.join(os.path.expanduser("~"), "data/raw_MRI/sample")
    # ROOT_DIR_TRAIN = os.path.join(os.path.expanduser("~"), "data/HGG")
    # train_dataset = MRIDatasetNumpySlices([ROOT_DIR_TRAIN])
    # validation_dataset = MRIDatasetNumpySlices([ROOT_DIR_VAL])
    # trainloader = DataLoader(train_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)
    # valloader = DataLoader(validation_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)

    train_dataset = MRIDatasetNumpySlices(train_directories)
    validation_dataset = MRIDatasetNumpySlices(validation_directories)
    trainloader = DataLoader(train_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True,
                             num_workers=config_train.NUM_WORKERS,
                             pin_memory=True
                             )
    valloader = DataLoader(validation_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True,
                           num_workers=config_train.NUM_WORKERS,
                           pin_memory=True
                           )

    unet = UNet().to(config_train.DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)

    train(unet,
          trainloader,
          valloader,
          optimizer,
          epochs=config_train.N_EPOCHS_CLIENT,
          filename="model.pth",
          plots_dir="predictions")
