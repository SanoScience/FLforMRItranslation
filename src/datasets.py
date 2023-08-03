import logging
import os
from glob import glob
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from configs import config_train
from src.files_operations import load_nii_slices, get_nii_filepaths, TransformNIIDataToNumpySlices


class MRIDatasetNII(Dataset):
    MIN_SLICE_INDEX = 50
    MAX_SLICE_INDEX = 125
    STEP = 1

    def __init__(self, data_dir: str,  t1_filepath_from_data_dir, t2_filepath_from_data_dir, transform: Compose, transform_order=None, n_patients=1, t1_to_t2=True):
        self.data_dir = data_dir
        self.transform = transform

        if t1_to_t2:
            images_filepaths, targets_filepaths = get_nii_filepaths(data_dir, t1_filepath_from_data_dir, t2_filepath_from_data_dir, n_patients)
        else:
            targets_filepaths, images_filepaths = get_nii_filepaths(data_dir, t1_filepath_from_data_dir, t2_filepath_from_data_dir, n_patients)

        self.images, self.targets = [], []

        for img_path in images_filepaths:
            slices, _, _ = load_nii_slices(img_path, transform_order, self.MIN_SLICE_INDEX, self.MAX_SLICE_INDEX)
            self.images.extend(slices)

        for target_path in targets_filepaths:
            slices, _, _ = load_nii_slices(target_path, transform_order, self.MIN_SLICE_INDEX, self.MAX_SLICE_INDEX)
            self.targets.extend(slices)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(np.expand_dims(self.images[index], axis=0))
        target = torch.from_numpy(np.expand_dims(self.targets[index], axis=0))

        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)

        return image.float(), target.float()


class MRIDatasetNumpySlices(Dataset):
    """
    Dataset class with previous use of TransformNIIDataToNumpySlices
    """
    EPS = 1e-6

    def __init__(self, data_dirs: List[str], t1_to_t2=True, normalize=True):
        if not isinstance(data_dirs, List):
            raise TypeError(f"Give parameter data_dirs: {data_dirs} is type: {type(data_dirs)} and should be list of string.")
        if t1_to_t2:
            image_type = "t1"
            target_type = "t2"
        else:
            image_type = "t2"
            target_type = "t1"

        self.normalize = normalize
        self.images = []
        self.targets = []
        for data_directory in data_dirs:
            self.images.extend(sorted(glob(f"{data_directory}/{image_type}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}")))
            self.targets.extend(sorted(glob(f"{data_directory}/{target_type}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}")))

            if len(self.images) == 0:
                raise FileNotFoundError(f"In directory {data_directory} no 't1' and 't2' directories found.")

    def __len__(self):
        return len(self.images)

    def _normalize(self, tensor: torch.Tensor):
        max_value = torch.max(tensor).data

        if max_value < self.EPS:
            raise ZeroDivisionError
        else:
            return tensor / max_value

    def __getitem__(self, index):
        image_path = self.images[index]
        target_path = self.targets[index]

        np_image = np.load(image_path)
        np_target = np.load(target_path)

        tensor_image = torch.from_numpy(np.expand_dims(np_image, axis=0))
        tensor_target = torch.from_numpy(np.expand_dims(np_target, axis=0))

        if self.normalize:
            try:
                tensor_image = self._normalize(tensor_image)
            except ZeroDivisionError:
                logging.warning(f"Data slice from the file {image_path} is a null image. "
                                f"All the values of the numpy array are 0.0")

            try:
                tensor_target = self._normalize(tensor_target)
            except ZeroDivisionError:
                logging.warning(f"Data slice from the file {target_path} is a null image. "
                                f"All the values of the numpy array are 0.0")

        # converting to float to be able to perform tensor multiplication
        # otherwise an error
        return tensor_image.float(), tensor_target.float()


def load_data(data_dir, batch_size, with_num_workers=True):
    # TODO: maybe input the test_dir instead of loader
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    val_dir = os.path.join(data_dir, "validation")

    trainset = MRIDatasetNumpySlices([train_dir])
    testset = MRIDatasetNumpySlices([test_dir])
    validationset = MRIDatasetNumpySlices([val_dir])

    if with_num_workers:
        # TODO: consider persistent_workers=True
        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=config_train.NUM_WORKERS,
                                  pin_memory=True, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=config_train.NUM_WORKERS,
                                 pin_memory=True, shuffle=True)
        val_loader = DataLoader(validationset, batch_size=batch_size, num_workers=config_train.NUM_WORKERS,
                                pin_memory=True, shuffle=True)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validationset, batch_size=batch_size,  shuffle=True)

    return train_loader, test_loader, val_loader
