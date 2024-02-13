import sys
import numpy as np
from os import path, listdir
import torch

from configs import config_train
from src.visualization import plot_batch
from src.datasets import MRIDatasetNumpySlices

from torch.utils.data import DataLoader

if __name__ == '__main__':
    ROOT_TRAIN_DIR = path.join(path.expanduser("~"), "")
    data_dir_ares = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg/validation"

    if len(sys.argv) > 1:
        data_dir_ares = sys.argv[1]

    dataset = MRIDatasetNumpySlices([data_dir_ares], translation_direction=config_train.TRANSLATION)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    first_iteration = True
    for images, targets in dataloader:
        for image, target in zip(images, targets):
            if first_iteration:
                image_shape = image.shape
                target_shape = target.shape
                first_iteration = False
            else:
                if image.shape != image_shape:
                    print(f"WARNING: The shapes are different", image_shape, "!=", image.shape)
                elif target.shape != target_shape:
                    print(f"WARNING: The shapes are different", target_shape, "!=", target.shape)
                else:
                    print("good")

    plot_batch([images, targets], filepath="plot_maybe_bad.jpg", show=False)



