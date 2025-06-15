"""
Dataset implementations for MRI image processing.
Provides classes for handling slice-based and volume-based MRI data.
"""

import logging
import os
from glob import glob
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.utils.files_operations import load_nii_slices, get_nii_filepaths, TransformNIIDataToNumpySlices, trim_image


class MRIDatasetNumpySlices(Dataset):
    """Dataset class for MRI slices stored as numpy arrays.
    
    Attributes:
        normalize: Whether to normalize images to [0,1]
        binarize: Whether to binarize target images
        squeeze: Whether to squeeze singleton dimensions
        metric_mask: Whether metric masks are available
        image_size: Optional target size for images

    Expects the following directory structure:
    data_dir/
        ├── t1/
        │   ├── subject1_slice1.npy
        │   ├── subject1_slice2.npy
        │   └── ...
        ├── t2/
        │   ├── subject1_slice1.npy
        │   ├── subject1_slice2.npy
        │   └── ...
        └── metric_masks/ (optional)
            ├── subject1_slice1.npy
            ├── subject1_slice2.npy
            └── ...
    """
    EPS = 1e-6

    def __init__(self, data_dir: Union[str, List[str]], 
                 translation_direction: Optional[Tuple[enums.ImageModality, enums.ImageModality]] = None,
                 target_dir: Optional[str] = None, 
                 image_size: Optional[Tuple[int, int]] = None,
                 normalize: bool = True,
                 binarize: bool = False,
                 metric_mask_dir: Optional[str] = None,
                 squeeze: bool = False,
                 input_target_set_union: bool = False) -> None:

        # declaring booleans
        self.normalize = normalize
        self.binarize = binarize
        self.squeeze = squeeze
        self.metric_mask = metric_mask_dir is not None

        self.image_size = image_size

        # declaring path lists
        self.images = []
        self.targets = []
        self.masks_for_metrics = []

        if translation_direction:
            if target_dir:
                raise ValueError("Either `translation_direction` or `target_dir` has to be specified, NOT BOTH.")
            if isinstance(translation_direction, Tuple):
                if len(translation_direction) == 2:
                    # the translation is a two element tuple
                    # the first element is the input
                    # the second element is the target
                    # e.g. (ImageModality.T1, ImageModality.T2)
                    # stands for T1 -> T2 translation
                    # it is directly transferred to the predefined file structure
                    # see files_operations.TransformNIIDataToNumpySlices
                    image_type = translation_direction[0].name.lower()
                    target_type = translation_direction[1].name.lower()
                else:
                    raise ValueError(
                        "The 'translation_direction' should be a 2-element tuple, with the first element with input "
                        "and the second the target of the translation e.g. (ImageModality.T1, ImageModality.T2)")
            else:
                raise TypeError(f"Given parameter 'translation_direction': {translation_direction} "
                                f"is type: {type(translation_direction)} and should be a tuple")
            if isinstance(data_dir, List):
                for data_directory in data_dir:
                    self.images.extend(
                        sorted(glob(f"{data_directory}/{image_type}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}")))
                    self.targets.extend(
                        sorted(glob(f"{data_directory}/{target_type}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}")))
                    if metric_mask_dir:
                        self.masks_for_metrics.extend(
                            sorted(glob(
                                f"{data_directory}/{metric_mask_dir}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}")))
            else:
                image_path_like = f"{data_dir}/{image_type}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}"
                target_path_like = f"{data_dir}/{target_type}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}"
                print(f"Loading network input data from path like: {image_path_like}")
                print(f"Loading network output data from path like: {target_path_like}")
                self.images = sorted(glob(image_path_like))
                self.targets = sorted(glob(target_path_like))

                if metric_mask_dir:
                    self.masks_for_metrics = sorted(glob(f"{data_dir}/{metric_mask_dir}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}"))

        elif target_dir:
            self.images = sorted(glob(f"{data_dir}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}"))
            self.targets = sorted(glob(f"{target_dir}/*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}"))

        else:
            raise ValueError("You either `translation_direction` or `target_dir` has to be specified.")

        if len(self.images) == 0 or len(self.targets) == 0:
            raise FileNotFoundError(f"In directory {data_dir} no provided inputs or targets directories found.\n",
                                    f"Check {translation_direction} and the directory names in the provided directory")  
        
        if input_target_set_union:
            self.images, self.targets = self._filepath_list_union(self.images, self.targets)
            if len(self.images) == 0 or len(self.targets) == 0:
                raise FileNotFoundError(f"The given directories have no common file names. The union resulted in an empty lists.")
            
        # print(f"Input paths: ", *self.images)
        # print(f"Output paths: ", *self.targets)


    @staticmethod
    def _filepath_list_union(list1, list2):
        # Extract filenames from the filepaths in both lists
        filenames1 = {fp.split(os.path.sep)[-1] for fp in list1}
        filenames2 = {fp.split(os.path.sep)[-1] for fp in list2}

        # Find the common filenames
        common_filenames = filenames1.intersection(filenames2)

        print("Excluded number filepaths for inputs: ", len([fp for fp in list1 if fp.split(os.path.sep)[-1] not in common_filenames]))
        print("Excluded number filepaths for targets: ", len([fp for fp in list2 if fp.split(os.path.sep)[-1] not in common_filenames]))

        return [fp for fp in list1 if fp.split(os.path.sep)[-1] in common_filenames], [fp for fp in list2 if fp.split(os.path.sep)[-1] in common_filenames]
    
    def __len__(self):
        return len(self.images)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        max_value = torch.max(tensor).data

        if max_value < self.EPS:
            raise ZeroDivisionError
        else:
            return tensor / max_value

    def _trim_image(self, image):
        return trim_image(image, self.image_size)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                              Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        image_path = self.images[index]
        target_path = self.targets[index]

        np_image = np.load(image_path)
        np_target = np.load(target_path)

        if self.squeeze:
            np_image = np_image[0]

        if self.image_size is not None:
            np_image = self._trim_image(np_image)
            np_target = self._trim_image(np_target)

        tensor_image = torch.from_numpy(np.expand_dims(np_image, axis=0))
        tensor_target = torch.from_numpy(np.expand_dims(np_target, axis=0))

        if self.binarize:
            tensor_target = tensor_target > 0
            tensor_target = tensor_target.int()

        if self.normalize:
            try:
                tensor_image = self._normalize(tensor_image)
            except ZeroDivisionError:
                logging.warning(f"Data slice from the file {image_path} is a null image. "
                                f"All the values of the numpy array are 0.0")
            if not self.binarize:
                try:
                    tensor_target = self._normalize(tensor_target)
                    tensor_target = tensor_target.float()
                except ZeroDivisionError:
                    logging.warning(f"Data slice from the file {target_path} is a null image. "
                                    f"All the values of the numpy array are 0.0")
                    
        if self.metric_mask:
            # all the same as previously
            mask_path = self.masks_for_metrics[index]
            np_mask = np.load(mask_path)
            np_mask = np_mask > 0  # binarize
            tensor_mask = torch.from_numpy(np.expand_dims(np_mask, axis=0))
            return tensor_image.float(), tensor_target, tensor_mask[0].int()

        # converting to float to be able to perform tensor multiplication
        # otherwise an error
        return tensor_image.float(), tensor_target.float()


class VolumeEvaluation(Dataset):
    """Dataset for evaluating 3D volumes of predictions against ground truth."""

    def __init__(self, ground_truth_path: str, predicted_path: str, 
                 squeeze_pred: bool = True) -> None:
        self.ground_truth_path = ground_truth_path
        self.predicted_path = predicted_path
        self.squeeze_pred = squeeze_pred

        patient_ids = [files.split('-')[1] for files in os.listdir(ground_truth_path)]   # the second part of the file is the patients id
        self.patient_ids = list(set(patient_ids))

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        patient_id = self.patient_ids[index]

        print(f"Patient id: {patient_id}")

        target_img_paths = glob(os.path.join(self.ground_truth_path, f"*{patient_id}*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}"))
        predicted_img_paths = glob(os.path.join(self.predicted_path, f"*{patient_id}*{TransformNIIDataToNumpySlices.SLICES_FILE_FORMAT}"))

        target_slices = [np.load(fp) for fp in target_img_paths]
        predicted_slices = [np.load(fp) for fp in predicted_img_paths]

        if self.squeeze_pred:
            predicted_slices = [pred[0] for pred in predicted_slices]

        target_volume = np.stack(target_slices)
        predicted_volume = np.stack(predicted_slices)

        target_volume = target_volume > 0  # binarize

        return target_volume, predicted_volume
    

class MRIDatasetNII(Dataset):  # NOT USED
    """
    A dataset which loads full 3D images .nii images to memory. Might be less efficient and limits some shuffling
    possibilities which is important in training on smaller datasets.
    Therefore, often utilized pipeline was:
    1. TransformNIIDataToNumpySlices to transform the dataset into 2D slices
    2. Then MRIDatasetNumpySlices used as the torch.Dataset to load the data efficiently
    """
    MIN_SLICE_INDEX = 50
    MAX_SLICE_INDEX = 125
    STEP = 1

    def __init__(self, data_dir: str, t1_filepath_from_data_dir, t2_filepath_from_data_dir, transform: Compose,
                 image_size=None, transform_order=None, n_patients=1, t1_to_t2=True):
        self.data_dir = data_dir
        self.transform = transform

        if t1_to_t2:
            images_filepaths, targets_filepaths = get_nii_filepaths(data_dir, t1_filepath_from_data_dir,
                                                                    t2_filepath_from_data_dir, n_patients)
        else:
            targets_filepaths, images_filepaths = get_nii_filepaths(data_dir, t1_filepath_from_data_dir,
                                                                    t2_filepath_from_data_dir, n_patients)

        self.images, self.targets = [], []

        for img_path in images_filepaths:
            slices, _, _ = load_nii_slices(img_path, transform_order, image_size, self.MIN_SLICE_INDEX,
                                           self.MAX_SLICE_INDEX)
            self.images.extend(slices)

        for target_path in targets_filepaths:
            slices, _, _ = load_nii_slices(target_path, transform_order, image_size, self.MIN_SLICE_INDEX,
                                           self.MAX_SLICE_INDEX)
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
