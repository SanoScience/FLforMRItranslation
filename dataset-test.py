import gzip
import nibabel as nib

import os
from os import path
import torch
import matplotlib.pyplot as plt
import numpy as np

from common.utils import MinMaxScalar, plot_batch
from common import config_train
from PIL import Image
from common.datasets import MRIDatasetNumpySlices, get_optimal_slice_range

from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader


ROOT_TRAIN_DIR = path.join(path.expanduser("~"), "data/hgg_transformed/train")
data_dir_ares = 

dataset = MRIDatasetNumpySlices([ROOT_TRAIN_DIR])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

images, targets = next(iter(dataloader))

plot_batch(images, targets)
