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
    # current_data_dir = "C:\\Users\\JanFiszer\\data\\HGG\\"
    # target_root_dir = "C:\\Users\\JanFiszer\\data\\hgg_transformed\\"

    transformer = TransformDataset(target_root_dir, current_data_dir)
    transformer.create_train_val_test_sets(n_patients=n_patients, seed=42)

