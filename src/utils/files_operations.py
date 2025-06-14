"""
File operations for MRI data preprocessing and management.
Handles NIfTI file loading, slice extraction, and dataset organization.
"""

import logging
import os
import random
import re
import traceback
from glob import glob
from typing import Tuple, Optional, List

from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


class TransformNIIDataToNumpySlices:
    """
    Transform 3D NIfTI files into 2D numpy slices for training.

    Attributes:
        target_root_dir: Root directory for output
        origin_data_dir: Source directory containing NIfTI files
        transpose_order: Order of axes for transposing images
        target_zero_ratio: Minimum non-zero ratio for valid slices
        image_size: Optional target size for output images

    Resulting directory structure:
    target_root_dir/
    ├── train/
    │   ├── t1/
    │   │   ├── subject1_slice1.npy
    │   │   ├── subject1_slice2.npy
    │   │   └── ...
    │   ├── t2/
    │   │   ├── subject1_slice1.npy
    │   │   ├── subject1_slice2.npy
    │   │   └── ...
    │   └── flair/ (if applicable)
    """

    MIN_SLICE_INDEX = -1
    MAX_SLICE_INDEX = -1
    SLICES_FILE_FORMAT = ".npy"
    DIVISION_SETS = ["train", "test", "validation"]

    def __init__(self, target_root_dir: str,
                 origin_data_dir: str,
                 transpose_order: Tuple[int, ...],
                 target_zero_ratio: float = 0.9,
                 image_size: Optional[Tuple[int, int]] = None,
                 leave_patient_name: bool = True) -> None:
        self.target_root_dir = target_root_dir
        self.origin_data_dir = origin_data_dir
        self.transpose_order = transpose_order
        self.target_zero_ratio = target_zero_ratio
        self.image_size = image_size
        self.leave_patient_name = leave_patient_name

    def create_empty_dirs(self, flair: bool = False) -> None:
        """Create directory structure for dataset splits."""
        # creating utilized directories
        images_types = ["t1", "t2"]

        if flair:
            images_types.append("flair")

        for s in self.DIVISION_SETS:
            set_dir = os.path.join(self.target_root_dir, s)
            try_create_dir(set_dir)
            for image_type in images_types:
                full_dir_name = os.path.join(set_dir, image_type)
                try_create_dir(full_dir_name, allow_overwrite=False)

    def create_train_val_test_sets(self,
                                   t1_filepath_from_data_dir,
                                   t2_filepath_from_data_dir,
                                   flair_filepath_from_data_dir=None,
                                   train_size=0.75,
                                   n_patients=-1,
                                   validation_size=0.1):
        # creating target directory if already exists.
        try_create_dir(self.target_root_dir)
        # creating inner directories
        # in each of the returned lists (train, test, val)
        # the order goes as follows: t1, t2, flair
        self.create_empty_dirs(flair=flair_filepath_from_data_dir is not None)

        # loading the data
        t1_filepaths, t2_filepaths, flair_filepaths = get_nii_filepaths(self.origin_data_dir,
                                                                        t1_filepath_from_data_dir,
                                                                        t2_filepath_from_data_dir,
                                                                        flair_filepath_from_data_dir,
                                                                        n_patients)

        # splitting filenames into train and test sets
        n_samples = len(t1_filepaths)
        n_train_samples = int(train_size * n_samples)
        n_val_samples = int(validation_size * n_samples)

        if n_val_samples <= 0:
            logging.warning(
                f"Validation set would be empty so the train set gonna be reduced.\nInput train_size: {train_size} validation_size: {validation_size}")
            n_val_samples = 1
            n_train_samples -= 1

        for s in self.DIVISION_SETS:
            if s == "train":
                lower_bound = 0
                upper_bound = n_train_samples
            elif s == "test":
                lower_bound = n_val_samples + n_train_samples
                upper_bound = n_samples + 1
            else:
                lower_bound = n_train_samples
                upper_bound = n_val_samples + n_train_samples

            current_set_filepaths_t1 = t1_filepaths[lower_bound:upper_bound]
            current_set_filepaths_t2 = t2_filepaths[lower_bound:upper_bound]
            current_set_filepaths_flair = flair_filepaths[lower_bound:upper_bound]

            self.create_set(current_set_filepaths_t1, current_set_filepaths_t2, current_set_filepaths_flair, s)

        print(f"\nSUCCESS\nCreated train and test directories in {self.target_root_dir} "
              f"from {n_train_samples} train, {n_val_samples} validation and {n_samples - n_train_samples - n_val_samples} "
              f"test 3D MRI images")

    def create_set(self, t1_paths, t2_paths, flair_paths, set_type_name):
        flair = len(flair_paths) > 0
        main_dir = os.path.join(self.target_root_dir, set_type_name)
        for index in range(len(t1_paths)):
            print(f"Files processed {t1_paths[index]}, {t2_paths[index]}, {flair_paths[index]}")
            print("Patient number ", index, " in process ...\n")

            t1_slices, taken_indices = load_nii_slices(t1_paths[index],
                                                                          self.transpose_order,
                                                                          self.image_size,
                                                                          self.MIN_SLICE_INDEX,
                                                                          self.MAX_SLICE_INDEX,
                                                                          target_zero_ratio=self.target_zero_ratio)

            min_slice_index = min(taken_indices)
            max_slice_index = max(taken_indices)

            if t1_slices:
                t2_slices, _, = load_nii_slices(t2_paths[index], self.transpose_order, self.image_size,
                                                  min_slice_index, max_slice_index)
                if flair:
                    flair_slices, _ = load_nii_slices(flair_paths[index], self.transpose_order, self.image_size,
                                                         min_slice_index, max_slice_index)

                for slice_index in range(len(t1_slices)):
                    filepath_dirs = t1_paths[index].split(os.path.sep)
                    if self.leave_patient_name:
                        filename = f"patient-{filepath_dirs[-1][:-7]}-slice{min_slice_index + slice_index}{self.SLICES_FILE_FORMAT}"
                    else:
                        filename = f"patient-{index}-slice{min_slice_index + slice_index}{self.SLICES_FILE_FORMAT}"

                    # saving a t1 slice
                    t1_slice_path = os.path.join(main_dir, "t1", filename)
                    np.save(t1_slice_path, t1_slices[slice_index])

                    # saving a t2 slice
                    t2_slice_path = os.path.join(main_dir, "t2", filename)
                    np.save(t2_slice_path, t2_slices[slice_index])

                    if flair:
                        # saving a t2 slice
                        flair_slice_path = os.path.join(main_dir, "flair", filename)
                        np.save(flair_slice_path, flair_slices[slice_index])
                        print("Created pair of t1, t2 and flair slices: ", t1_slice_path, t2_slice_path,
                              flair_slice_path)
                    else:
                        print("Created pair of t1 and t2 slices: ", t1_slice_path, t2_slice_path)

                if flair:
                    print(
                        f"T1, T2 and FLAIR slice shape {t1_slices[0].shape} {t2_slices[0].shape} {flair_slices[0].shape}")
                else:
                    print(f"T1 and T2 slice shape{t1_slices[0].shape} {t2_slices[0].shape}")
            else:
                print("Skipped due to the shape\n")


def trim_image(image: np.ndarray,
               target_image_size: Tuple[int, int]) -> np.ndarray:
    """
    Trim image to target size by removing equal margins from all sides.

    Args:
        image: Input image array
        target_image_size: Desired output dimensions

    Returns:
        Trimmed image array

    Raises:
        ValueError: If target size is larger than input size
    """
    x_pixels_margin = int((image.shape[0] - target_image_size[0]) / 2)
    y_pixels_margin = int((image.shape[1] - target_image_size[1]) / 2)

    if x_pixels_margin < 0 or y_pixels_margin < 0:
        raise ValueError(f"Target image size: {target_image_size} greater than original image size {image.shape}")

    return image[x_pixels_margin:target_image_size[0] + x_pixels_margin,
           y_pixels_margin:target_image_size[1] + y_pixels_margin]


def load_nii_slices(filepath: str, transpose_order, image_size: Optional[Tuple[int, int]] = None, min_slice_index=-1,
                    max_slices_index=-1, index_step=1, target_zero_ratio=0.9):
    def get_optimal_slice_range(brain_slices, eps=1e-4, target_zero_ratio=0.9):
        zero_ratios = np.array([np.sum(brain_slice < eps) / (brain_slice.shape[0] * brain_slice.shape[1])
                                for brain_slice in brain_slices])
        satisfying_given_ratio = np.where(zero_ratios < target_zero_ratio)[0]

        return satisfying_given_ratio

    # noinspection PyUnresolvedReferences
    img = nib.load(filepath).get_fdata()

    if max_slices_index > img.shape[-1]:  # img.shape[-1] == total number of slices
        raise ValueError

    # in case of brain image being in wrong shape
    # we want (n_slice, img_H, img_W)
    # it changes from (img_H, img_W, n_slices) to desired length
    if transpose_order is not None:
        img = np.transpose(img, transpose_order)

    if image_size is not None:
        img = [trim_image(brain_slice, image_size) for brain_slice in img]

    if min_slice_index == -1 or max_slices_index == -1:
        taken_indices = get_optimal_slice_range(img, target_zero_ratio=target_zero_ratio)
        print(f"Slice range used for file {filepath}: {taken_indices}")
    else:
        print(f"Slice range used for file {filepath}: <{min_slice_index, max_slices_index}>")
        taken_indices = range(min_slice_index, max_slices_index)

    selected_slices = [img[slice_index] for slice_index in taken_indices]

    return selected_slices, taken_indices


def get_nii_filepaths(data_dir, t1_filepath_from_data_dir, t2_filepath_from_data_dir, flair_filepath_from_data_dir,
                      n_patients=-1):
    # creating the t1 and t2 filepaths
    t1_filepaths = []
    t2_filepaths = []
    flair_filepaths = []

    local_dirs = os.listdir(data_dir)

    # if not specified taking all patients
    if n_patients == -1:
        n_patients = len(local_dirs)

    i = 0
    for local_dir in local_dirs:
        if i >= n_patients:
            # loop runs until
            # all directories are visited (for ends)
            # the number of patients is fulfilled (i >= n_patients)
            break
        # just for one dataset purposes
        # inside_dir = local_dirs[i].split('_')[0]
        if flair_filepath_from_data_dir is not None:  # if the path is specified we look for flair filepaths
            flair_like_path = os.path.join(data_dir, local_dir, flair_filepath_from_data_dir)
            flair_filepath = sorted(glob(flair_like_path))
            if len(flair_filepath) == 0:  # if not any found the directory is skipped (T1 and T2 also omitted)
                continue
            # for some dataset (e.g. oasis) we have multiple flair images for one patient
            # then we take just the last one `[-1]`
            # assumption: it doesn't matter which one we take (so we can take the last one)
            flair_filepaths.append(sorted(glob(flair_like_path))[-1])

        t1_like_path = os.path.join(data_dir, local_dir, t1_filepath_from_data_dir)
        t2_like_path = os.path.join(data_dir, local_dir, t2_filepath_from_data_dir)

        t1_filepaths.extend(sorted(glob(t1_like_path)))
        t2_filepaths.extend(sorted(glob(t2_like_path)))

        i += 1

    local_dirs_string = '\n'.join([loc_dir for loc_dir in local_dirs])

    print(f"Found {len(t1_filepaths)} t1 {len(t2_filepaths)} t2 and {len(flair_filepaths)} flair files. In files: "
          f"{local_dirs_string}")

    return t1_filepaths, t2_filepaths, flair_filepaths


def try_create_dir(dir_name, allow_overwrite=True):
    """
    Try to create a directory. If it exists, handle according to allow_overwrite flag.

    Args:
        dir_name: Directory path to create
        allow_overwrite: Flag indicating whether to overwrite existing directory

    Raises:
        FileExistsError: If the directory exists and allow_overwrite is False
        FileNotFoundError: If a part of the directory path does not exist
    """
    try:
        Path(dir_name).mkdir(parents=True, exist_ok=allow_overwrite)
    except FileExistsError:
        if allow_overwrite:
            logging.warning(
                f"Directory {dir_name} already exists. You may overwrite your files or create some collisions!")
        else:
            raise FileExistsError(
                f"Directory {dir_name} already exists. If you want to overwrite it change allow_overwrite for True")

    except FileNotFoundError:
        ex = FileNotFoundError(
            f"The path {dir_name} to directory willing to be created doesn't exist. You are in {os.getcwd()}.")

        traceback.print_exception(FileNotFoundError, ex, ex.__traceback__)


def create_segmentation_mask_dir(preprocess_dir_name, mask_dir_name, transpose_order,
                                 new_masked_dir_name="mask", indir_reading_name="t1", mask_fingerprint="*seg.nii.gz",
                                 output_format=".npy",
                                 only_with_glioma=False):
    """
    Create a directory of segmentation masks from NIfTI files.

    Args:
        preprocess_dir_name: Directory where preprocessed data is stored
        mask_dir_name: Directory containing the original mask NIfTI files
        transpose_order: Order of axes for transposing images
        new_masked_dir_name: Name of the new directory to create
        indir_reading_name: Subdirectory name for reading input data
        mask_fingerprint: Filename pattern to identify mask files
        output_format: File format for saving masks
        only_with_glioma: Flag to include only masks with glioma
    """
    mask_dirs = os.listdir(mask_dir_name)

    dir_path = os.path.join(preprocess_dir_name, indir_reading_name)
    print(f"Reading slices from directory: {dir_path}\n\n")

    patients_slices = get_brains_slices_info(dir_path)

    created_mask_dir = os.path.join(preprocess_dir_name, new_masked_dir_name)
    try_create_dir(created_mask_dir)

    print("Patients id and the slices range:")
    for patient_id, slices_range in patients_slices.items():
        print("Patient id: ", patient_id)
        print("Range: ", slices_range, "\n")

        print("Folder found:")
        full_dir_name = None
        for mask_dir in mask_dirs:
            if "_".join(
                    patient_id.split('_')[:-1]) in mask_dir:  # the patient ID has a "leftover" (.._t1) which is skipped
                print(mask_dir)
                full_dir_name = mask_dir
                break

        path_mask = os.path.join(mask_dir_name, full_dir_name)
        like_path_mask = os.path.join(path_mask, mask_fingerprint)
        mask_file = glob(like_path_mask)[0]
        mask_filepath = os.path.join(path_mask, mask_file)

        print(f"Used filepath to find the mask: {mask_filepath}")

        if only_with_glioma:
            # resetting found slices indices and to take only the one which have some glioma
            # target_zero_ratio ensures that we take only with at least one pixel
            slices_range = (-1, -1)
            target_zero_ratio = 1.0  # excluding full zeros
        else:
            target_zero_ratio = None  # not caring about the zero percentage, not considered anyway when we provide range slices

        mask_slices, utilized_slices = load_nii_slices(mask_filepath,
                                                       transpose_order,
                                                       min_slice_index=slices_range[0],
                                                       max_slices_index=slices_range[1],
                                                       target_zero_ratio=target_zero_ratio)  # not caring about the zero percentage

        for mask, mask_real_index in zip(mask_slices, utilized_slices):
            mask_slice_filename = f"patient-{patient_id}-slice{mask_real_index}{output_format}"
            mask_slice_filepath = os.path.join(created_mask_dir, mask_slice_filename)

            np.save(mask_slice_filepath, mask)
            print(f"Mask file: {mask_slice_filepath} saved.")


def get_brains_slices_info(dir_name):
    filenames = os.listdir(dir_name)
    patient_slices = {}

    # the file is in the format e.g. patient-Brats18_TCIA10_420_1_t1-slice108.npy
    # we are extracting ID which is always between "-"
    # in this case Brats18_TCIA10_420_1_t1
    # list(set(...)) for extracting unique values
    patients_id = list(set([f.split('-')[1] for f in filenames]))

    for patient_id in patients_id:
        slices_nr = []
        for f in filenames:
            if patient_id in f:
                slices_nr.append(int(re.search(r'slice(\d+)', f).group(1)))

        patient_slices[patient_id] = (min(slices_nr), max(slices_nr))

    return patient_slices


def test_mask_in(img_name, img_dir, breakpoint=10, failed_dir="failed"):
    img = np.load(os.path.join(img_dir, img_name))

    if np.sum(img[:, :breakpoint]):
        plt.imshow(img > 0)
        plt.savefig(os.path.join(failed_dir, img_name))
        print("\n\nWRONG MASKS IN THE IMAGE: ", img_name)

        return False

    else:
        return True
