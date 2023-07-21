import gzip
import nibabel as nib

import os
from os import path
import torch
import matplotlib.pyplot as plt
import numpy as np

from common.utils import MinMaxScalar
from common import config_train
from PIL import Image
from common.datasets import MRIDatasetNumpySlices, get_optimal_slice_range

from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader


ROOT_DIR = path.join(path.expanduser("~"), "data\\raw_MRI\\train")
# transform = Compose([ToTensor(), MinMaxScalar()])

dataset = MRIDatasetNumpySlices(ROOT_DIR)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

l = len(dataloader)
ll = len(dataloader.dataset)
#
# images, masks = next(iter(dataloader))
# print(images[0])


img_path = "C:\\Users\\JanFiszer\\data\\HGG\\Brats18_CBICA_ABY_1\\Brats18_CBICA_ABY_1_t1.nii.gz"
img = nib.load(img_path)
brain_slices = np.transpose(img.get_fdata(), (2, 0, 1))
n_slices = img.shape[-1]

optimal = get_optimal_slice_range(brain_slices, target_zero_ratio=0.80)

print(optimal)