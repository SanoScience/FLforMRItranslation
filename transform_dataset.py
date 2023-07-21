import os
import sys

from common.datasets import create_train_val_test_sets
from os import path


if __name__ == '__main__':
    target_root_dir = sys.argv[1]
    current_data_dir = sys.argv[2]
    # current_data_dir = "C:\\Users\\JanFiszer\\data\\HGG\\"
    # target_root_dir = "C:\\Users\\JanFiszer\\data\\testttt\\"

    create_train_val_test_sets(target_root_dir, current_data_dir, seed=42)

