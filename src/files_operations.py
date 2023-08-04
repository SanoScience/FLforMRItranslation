import logging
import os
import random
import traceback
from glob import glob
from typing import Tuple

import nibabel as nib
import numpy as np


def try_create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        logging.warning(f"Directory {dir_name} already exists. You may overwrite your files or create some collisions!")

    except FileNotFoundError:
        ex = FileNotFoundError(f"The path {dir_name} to directory willing to be created doesn't exist. You are in {os.getcwd()}.")

        traceback.print_exception(FileNotFoundError, ex, ex.__traceback__)


def load_nii_slices(filepath: str, transpose_order, min_slice_index=-1, max_slices_index=-1, index_step=1, target_zero_ratio=0.9):
    def get_optimal_slice_range(brain_slices, eps=1e-4, target_zero_ratio=0.9):
        zero_ratios = np.array([np.sum(brain_slice < eps) / (brain_slice.shape[0] * brain_slice.shape[1])
                                for brain_slice in brain_slices])
        satisfying_given_ratio = np.where(zero_ratios < target_zero_ratio)[0]

        upper_bound = satisfying_given_ratio[0]
        lower_bound = satisfying_given_ratio[-1]

        return upper_bound, lower_bound

    img = nib.load(filepath).get_fdata()

    if max_slices_index > img.shape[-1]:  # img.shape[-1] == total number of slices
        raise ValueError

    # in case of brain image being in wrong shape
    # we want (n_slice, img_H, img_W)
    # it changes from (img_H, img_W, n_slices) to desired length
    if transpose_order is not None:
        img = np.transpose(img, transpose_order)

    if min_slice_index == -1 or max_slices_index == -1:
        min_slice_index, max_slices_index = get_optimal_slice_range(img, target_zero_ratio=target_zero_ratio)

    print(f"Slice range used for file {filepath}: <{min_slice_index}, {max_slices_index}>")

    return [img[slice_index] for slice_index in range(min_slice_index, max_slices_index + 1, index_step)], min_slice_index, max_slices_index


def get_nii_filepaths(data_dir, t1_filepath_from_data_dir, t2_filepath_from_data_dir, n_patients=-1):
    # creating the t1 and t2 filepaths
    t1_filepaths = []
    t2_filepaths = []

    local_dirs = os.listdir(data_dir)

    # if not specified taking all patients
    if n_patients == -1:
        n_patients = len(local_dirs)

    for i in range(n_patients):
        # just for one dataset purposes
        # inside_dir = local_dirs[i].split('_')[0]

        t1_like_path = os.path.join(data_dir, local_dirs[i], t1_filepath_from_data_dir)
        t2_like_path = os.path.join(data_dir, local_dirs[i], t2_filepath_from_data_dir)

        t1_filepaths.extend(sorted(glob(t1_like_path)))
        t2_filepaths.extend(sorted(glob(t2_like_path)))

    local_dirs_string = '\n'.join([loc_dir for loc_dir in local_dirs])

    print(f"Found {len(t1_filepaths)} t1 files and {len(t2_filepaths)} t2 files. In files: "
          f"{local_dirs_string}")

    return t1_filepaths, t2_filepaths


class TransformNIIDataToNumpySlices:
    # by investigation in eda.ipynb obtained
    # MIN_SLICE_INDEX = 50
    # MAX_SLICE_INDEX = 125
    MIN_SLICE_INDEX = -1
    MAX_SLICE_INDEX = -1
    SLICES_FILE_FORMAT = ".npy"

    def __init__(self, target_root_dir: str, origin_data_dir: str, transpose_order: Tuple, target_zero_ratio=0.9):
        self.target_root_dir = target_root_dir
        self.origin_data_dir = origin_data_dir
        self.transpose_order = transpose_order
        self.target_zero_ratio = target_zero_ratio

    def create_empty_dirs(self):
        # TODO: deal with already created directories, to prevent overwriting, IDEA- patient_index + n_files or maybe not needed
        # creating utilized directories
        train_dir = os.path.join(self.target_root_dir, "train")
        test_dir = os.path.join(self.target_root_dir, "test")
        val_dir = os.path.join(self.target_root_dir, "validation")

        t1_train_dir = os.path.join(train_dir, "t1")
        t2_train_dir = os.path.join(train_dir, "t2")
        t1_test_dir = os.path.join(test_dir, "t1")
        t2_test_dir = os.path.join(test_dir, "t2")
        t1_val_dir = os.path.join(val_dir, "t1")
        t2_val_dir = os.path.join(val_dir, "t2")

        for directory in [train_dir, test_dir, val_dir,
                          t1_train_dir, t2_train_dir, t1_test_dir, t2_test_dir, t1_val_dir, t2_val_dir]:
            try_create_dir(directory)

        return t1_train_dir, t2_train_dir, t1_test_dir, t2_test_dir, t1_val_dir, t2_val_dir

    def create_train_val_test_sets(self,
                                   t1_filepath_from_data_dir,
                                   t2_filepath_from_data_dir,
                                   train_size=0.75,
                                   n_patients=-1,
                                   validation_size=0.1,
                                   seed=-1,
                                   shuffle=True):
        if seed != -1:
            random.seed(seed)
        # creating target directory if already exists.
        try_create_dir(self.target_root_dir)
        # creating inner directories
        t1_train_dir, t2_train_dir, t1_test_dir, t2_test_dir, t1_val_dir, t2_val_dir = self.create_empty_dirs()
        print("Created directories: ",
              t1_train_dir, t2_train_dir, t1_test_dir, t2_test_dir, t1_val_dir, t2_val_dir,
              "\n", sep='\n')

        # loading the data
        t1_filepaths, t2_filepaths = get_nii_filepaths(self.origin_data_dir,
                                                       t1_filepath_from_data_dir,
                                                       t2_filepath_from_data_dir,
                                                       n_patients)

        # splitting filenames into train and test sets
        # TODO: when patients < 10
        n_samples = len(t1_filepaths)
        n_train_samples = int(train_size * n_samples)
        n_val_samples = int(validation_size * n_samples)

        if n_val_samples <= 0:
            logging.warning(f"Validation set would be empty so the train set gonna be reduced.\nInput train_size: {train_size} validation_size: {validation_size}")
            n_val_samples = 1
            n_train_samples -= 1

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

        print("Creating train set...")
        self.create_set(t1_train_paths, t2_train_paths, t1_train_dir, t2_train_dir)

        print("Creating test set...")
        self.create_set(t1_test_paths, t2_test_paths, t1_test_dir, t2_test_dir)

        print("Creating validation set...")
        self.create_set(t1_val_paths, t2_val_paths, t1_val_dir, t2_val_dir)

        print(f"\nSUCCESS\nCreated train and test directories in {self.target_root_dir} "
              f"from {n_train_samples} train, {n_val_samples} validation and {n_samples - n_train_samples - n_val_samples} "
              f"test 3D MRI images")

    def create_set(self, t1_paths, t2_paths, t1_dir, t2_dir):
        for patient_id, (t1_path, t2_path) in enumerate(zip(t1_paths, t2_paths)):
            print("Patient number ", patient_id, " in process...\n")
            t1_slices, min_slice_index, max_slice_index = load_nii_slices(t1_path,
                                                                          self.transpose_order,
                                                                          self.MIN_SLICE_INDEX,
                                                                          self.MAX_SLICE_INDEX,
                                                                          target_zero_ratio=self.target_zero_ratio)
            t2_slices, _, _ = load_nii_slices(t2_path, self.transpose_order, min_slice_index, max_slice_index)

            for index, (t1_slice, t2_slice) in enumerate(zip(t1_slices, t2_slices)):
                filename = f"patient{patient_id}-slice{index}{self.SLICES_FILE_FORMAT}"
                # saving a t1 slice
                t1_slice_path = os.path.join(t1_dir, filename)
                np.save(t1_slice_path, t1_slice)

                # saving a t2 slice
                t2_slice_path = os.path.join(t2_dir, filename)
                np.save(t2_slice_path, t2_slice)

                print("Created pair of t1 and t2 slices: ", t1_slice_path, t2_slice_path)

            print(f"T1 and T2 slice shape{t1_slice.shape} {t2_slice.shape}")