import sys

from os import path
import torch

from src.visualization import plot_batch
from src.datasets import MRIDatasetNumpySlices

from torch.utils.data import DataLoader

if __name__ == '__main__':
    ROOT_TRAIN_DIR = path.join(path.expanduser("~"), "")
    data_dir_ares = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg/validation"

    if len(sys.argv) > 1:
        data_dir_ares = sys.argv[1]

    dataset = MRIDatasetNumpySlices([data_dir_ares])

    print(dataset.images)
    print(dataset.targets)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    images, targets = next(iter(dataloader))

    plot_batch([images, targets], filepath="plot_maybe_bad.jpg", show=False)