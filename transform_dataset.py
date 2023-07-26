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
    # current_data_dir = "C:\\Users\\JanFiszer\\data\\HCP_MGH"
    # target_root_dir = "C:\\Users\\JanFiszer\\data\\hcp_mgh_first_test\\"

    transpose_order = (2, 1, 0)
    transformer = TransformDataset(target_root_dir, current_data_dir, transpose_order)
    transformer.create_train_val_test_sets("anat\\T1\\T1_bet.nii.gz", "anat\\t2\\T2_bet_reg.nii.gz", seed=42)

