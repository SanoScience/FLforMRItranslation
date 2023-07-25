import logging
import os
import random
from glob import glob
from PIL import Image

import nibabel as nib
import numpy as np

from typing import List

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from common import config_train, utils

# TODO: split the functions into a class anther file or smth


class MRIDatasetNII(Dataset):
    MIN_SLICE_INDEX = 50
    MAX_SLICE_INDEX = 125
    STEP = 1

    def __init__(self, data_dir: str, transform: Compose, n_patients=1, t1_to_t2=True):
        self.data_dir = data_dir
        self.transform = transform

        if t1_to_t2:
            images_filepaths, targets_filepaths = get_nii_filepaths(data_dir, n_patients)
        else:
            targets_filepaths, images_filepaths = get_nii_filepaths(data_dir, n_patients)

        self.images, self.targets = [], []

        for img_path in images_filepaths:
            slices, _, _ = load_nii_slices(img_path, self.MIN_SLICE_INDEX, self.MAX_SLICE_INDEX)
            self.images.extend(slices)

        for target_path in targets_filepaths:
            slices, _, _ = load_nii_slices(target_path, self.MIN_SLICE_INDEX, self.MAX_SLICE_INDEX)
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
    Dataset class with previous use of create_training_directories()
    """
    EPS = 1e-6
    # by investigation in eda.ipynb obtained
    # MIN_SLICE_INDEX = 50
    # MAX_SLICE_INDEX = 125
    MIN_SLICE_INDEX = -1
    MAX_SLICE_INDEX = -1
    SLICES_FILE_FORMAT = ".npy"

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
            self.images.extend(glob(f"{data_directory}/{image_type}/*{self.SLICES_FILE_FORMAT}"))
            self.targets.extend(glob(f"{data_directory}/{target_type}/*{self.SLICES_FILE_FORMAT}"))

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


def load_nii_slices(filepath: str, min_slice_index=-1, max_slices_index=-1, index_step=1):
    img = nib.load(filepath).get_fdata()

    if max_slices_index > img.shape[-1]:  # img.shape[-1] == total number of slices
        raise ValueError

    if min_slice_index == -1 or max_slices_index == -1:
        min_slice_index, max_slices_index = get_optimal_slice_range(img)

    print(f"Slice range used for file {filepath}: <{min_slice_index}, {max_slices_index}>")

    return [img[:, :, slice_index] for slice_index in range(min_slice_index, max_slices_index + 1, index_step)], min_slice_index, max_slices_index


def get_nii_filepaths(data_dir, n_patients=-1):
    local_dirs = os.listdir(data_dir)

    # if not specified taking all patients
    if n_patients == -1:
        n_patients = len(local_dirs)

    # creating the t1 and t2 filepaths
    t1_filepaths = []
    t2_filepaths = []

    for i in range(n_patients):
        t1_filepaths.extend(sorted(glob(f"{data_dir}/{local_dirs[i]}/*t1.nii.gz")))
        t2_filepaths.extend(sorted(glob(f"{data_dir}/{local_dirs[i]}/*t2.nii.gz")))

    print(f"Found {len(t1_filepaths)} t1 files and {len(t2_filepaths)} t2 files")

    return t1_filepaths, t2_filepaths


def create_empty_dirs(root_dir: str):
    # TODO: deal with already created directories, to prevent overwriting, IDEA- patient_index + n_files or maybe not needed
    # creating utilized directories
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    val_dir = os.path.join(root_dir, "validation")

    t1_train_dir = os.path.join(train_dir, "t1")
    t2_train_dir = os.path.join(train_dir, "t2")
    t1_test_dir = os.path.join(test_dir, "t1")
    t2_test_dir = os.path.join(test_dir, "t2")
    t1_val_dir = os.path.join(val_dir, "t1")
    t2_val_dir = os.path.join(val_dir, "t2")

    for directory in [train_dir, test_dir, val_dir,
                      t1_train_dir, t2_train_dir, t1_test_dir, t2_test_dir, t1_val_dir, t2_val_dir]:
        utils.try_create_dir(directory)

    return t1_train_dir, t2_train_dir, t1_test_dir, t2_test_dir, t1_val_dir, t2_val_dir


def create_train_val_test_sets(target_root_dir: str,
                               origin_data_dir: str,
                               train_size=0.75,
                               n_patients=-1,
                               validation_size=0.1,
                               seed=-1,
                               shuffle=True):
    # TODO: let to add other dataset, but maybe not so important for FL
    if seed != -1:
        random.seed(seed)
    # creating target directory if already exists.
    utils.try_create_dir(target_root_dir)
    # creating inner directories
    t1_train_dir, t2_train_dir, t1_test_dir, t2_test_dir, t1_val_dir, t2_val_dir = create_empty_dirs(target_root_dir)
    print("Created directories: ",
          t1_train_dir, t2_train_dir, t1_test_dir, t2_test_dir, t1_val_dir, t2_val_dir,
          "\n", sep='\n')

    # loading the data
    t1_filepaths, t2_filepaths = get_nii_filepaths(origin_data_dir, n_patients)

    # splitting filenames into train and test sets
    n_samples = len(t1_filepaths)
    n_train_samples = int(train_size * n_samples)
    n_val_samples = int(validation_size * n_samples)

    if shuffle:
        filepaths = list(zip(t1_filepaths, t2_filepaths))
        random.shuffle(filepaths)
        t1_filepaths, t2_filepaths = zip(*filepaths)

    t1_train_paths = t1_filepaths[:n_train_samples]
    t1_val_paths = t1_filepaths[n_train_samples:n_val_samples + n_train_samples]
    t1_test_paths = t1_filepaths[n_val_samples + n_train_samples:]

    t2_train_paths = t2_filepaths[:n_train_samples]
    t2_val_paths = t2_filepaths[n_train_samples:n_val_samples + n_train_samples]
    t2_test_paths = t2_filepaths[n_val_samples + n_train_samples:]

    # creating train set
    print("Creating train set...")
    create_set(t1_train_paths, t2_train_paths, t1_train_dir, t2_train_dir)
    # for patient_id, (t1_path, t2_path) in enumerate(zip(t1_train_paths, t2_train_paths)):
    #     print("Patient number ", patient_id, " in process...\n")
    #
    #     t1_slices, min_slice_index, max_slice_index = load_nii_slices(t1_path, MRIDatasetNumpySlices.MIN_SLICE_INDEX, MRIDatasetNumpySlices.MAX_SLICE_INDEX)
    #     t2_slices, _, _ = load_nii_slices(t2_path, min_slice_index, max_slice_index)
    #
    #     for index, (t1_slice, t2_slice) in enumerate(zip(t1_slices, t2_slices)):
    #         # getting the origin directory e.g. Brats18_2013_2_1
    #         origin_dir = t1_path.split('/')[-2]
    #         filename = f"{origin_dir}-patient{patient_id}-slice{index}{MRIDatasetNumpySlices.SLICES_FILE_FORMAT}"
    #
    #         # saving a t1 slice
    #         t1_slice_path = os.path.join(t1_train_dir, filename)
    #         np.save(t1_slice_path, t1_slice)
    #
    #         # saving a t2 slice
    #         t2_slice_path = os.path.join(t2_train_dir, filename)
    #         np.save(t2_slice_path, t2_slice)
    #
    #         print("Created pair of t1 and t2 slices: ", t1_slice_path, t2_slice_path)

    # creating test set
    print("Creating test set...")
    create_set(t1_test_paths, t2_test_paths, t1_test_dir, t2_test_dir)
    # for patient_id, (t1_path, t2_path) in enumerate(zip(t1_test_paths, t2_test_paths)):
    #     print("Patient number ", patient_id, " in process...\n")
    #     t1_slices, min_slice_index, max_slice_index = load_nii_slices(t1_path, MRIDatasetNumpySlices.MIN_SLICE_INDEX, MRIDatasetNumpySlices.MAX_SLICE_INDEX)
    #     t2_slices, _, _ = load_nii_slices(t2_path, min_slice_index, max_slice_index)
    #
    #     for index, (t1_slice, t2_slice) in enumerate(zip(t1_slices, t2_slices)):
    #         filename = f"patient{patient_id}-slice{index}{MRIDatasetNumpySlices.SLICES_FILE_FORMAT}"
    #         # saving a t1 slice
    #         t1_slice_path = os.path.join(t1_test_dir, filename)
    #         np.save(t1_slice_path, t1_slice)
    #
    #         # saving a t2 slice
    #         t2_slice_path = os.path.join(t2_test_dir, filename)
    #         np.save(t2_slice_path, t2_slice)
    #
    #         print("Created a pair of t1 and t2 slices: ", t2_slice_path, t1_slice_path)

    # creating validation set
    print("Creating validation set...")
    create_set(t1_val_paths, t2_val_paths, t1_val_dir, t2_val_dir)

    print(f"\nSUCCESS\nCreated train and test directories in {target_root_dir} "
          f"from {n_train_samples} train, {n_val_samples} validation and {n_samples - n_train_samples - n_val_samples} "
          f"test 3D MRI images")


def create_set(t1_paths, t2_paths, t1_dir, t2_dir):
    for patient_id, (t1_path, t2_path) in enumerate(zip(t1_paths, t2_paths)):
        print("Patient number ", patient_id, " in process...\n")
        t1_slices, min_slice_index, max_slice_index = load_nii_slices(t1_path, MRIDatasetNumpySlices.MIN_SLICE_INDEX,
                                                                      MRIDatasetNumpySlices.MAX_SLICE_INDEX)
        t2_slices, _, _ = load_nii_slices(t2_path, min_slice_index, max_slice_index)

        for index, (t1_slice, t2_slice) in enumerate(zip(t1_slices, t2_slices)):
            filename = f"patient{patient_id}-slice{index}{MRIDatasetNumpySlices.SLICES_FILE_FORMAT}"
            # saving a t1 slice
            t1_slice_path = os.path.join(t1_dir, filename)
            np.save(t1_slice_path, t1_slice)

            # saving a t2 slice
            t2_slice_path = os.path.join(t2_dir, filename)
            np.save(t2_slice_path, t2_slice)

            print("Created pair of t1 and t2 slices: ", t1_slice_path, t2_slice_path)

# used when file format is jpeg
def save_as_img(array, path):
    img = Image.fromarray(array)
    img = img.convert('RGB')
    img.save(path)


def get_optimal_slice_range(brain_slices, eps=1e-4, target_zero_ratio=0.9):
    # in case of brain image being in wrong shape
    # we want (n_slice, img_H, img_W)
    # it changes from (img_H, img_W, n_slices) to desired length
    if brain_slices.shape[0] == config_train.IMAGE_SIZE[0]:
        brain_slices = np.transpose(brain_slices, (2, 0, 1))

    zero_ratios = np.array([np.sum(brain_slice < eps) / (brain_slice.shape[0] * brain_slice.shape[0])
                            for brain_slice in brain_slices])
    satisfying_given_ratio = np.where(zero_ratios < target_zero_ratio)[0]

    upper_bound = satisfying_given_ratio[0]
    lower_bound = satisfying_given_ratio[-1]

    return upper_bound, lower_bound







