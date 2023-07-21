import os

from common.datasets import create_train_test_sets
from os import path


def main():
    current_data_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/BraTS/HGG"
    target_root_dir = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/data"

    create_train_test_sets(target_root_dir, current_data_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
