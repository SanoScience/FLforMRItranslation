import os
import time

import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch

from common import datasets, models, config_train, utils

from client.utils import train

ROOT_DIR_TRAIN = os.path.join(os.path.expanduser("~"), "data/smart_slice_selection_MRI/sample")
train_dataset = datasets.MRIDatasetNumpySlices(ROOT_DIR_TRAIN)
# ROOT_DIR_TRAIN = os.path.join(os.path.expanduser("~"), "data/HGG")
# train_dataset = datasets.MRIDatasetNII(ROOT_DIR_TRAIN, transform=None)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)

unet = models.UNet().to(config_train.DEVICE)

optimizer = optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)

ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))

train(unet,
      trainloader,
      optimizer,
      epochs=config_train.N_EPOCHS_CLIENT,
      filename="model.pth")
