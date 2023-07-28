import os
import sys

from common.datasets import TransformDataset
from os import path


if __name__ == '__main__':
    target_root_dir = sys.argv[1]
    current_data_dir = sys.argv[2]
    if len(sys.argv) > 3:
        n_patients = int(sys.argv[3])
    else:
        n_patients = -1
    # current_data_dir = "C:\\Users\\JanFiszer\\data\\mgh_masks"
    # target_root_dir = "C:\\Users\\JanFiszer\\data\\mgh_masks_transformed"

    # C:\Users\JanFiszer\data\HCP_Wu-Minn\114318_3T_Structural_preproc\114318\T1w
    transpose_order = (2, 0, 1)
    transformer = TransformDataset(target_root_dir, current_data_dir, transpose_order, target_zero_ratio=0.7)
    transformer.create_train_val_test_sets("T1w/T1*restore_brain.nii.gz", "T1w/T2*restore_brain.nii.gz", n_patients=n_patients, seed=42)

