import os
import sys
from shutil import copy2

import torch.optim as optim

from src.ml.datasets import *
from src.ml.models import *

from torch.utils.data import DataLoader

if __name__ == '__main__':

    if config_train.LOCAL:
        train_directories = ["C:\\Users\\JanFiszer\\data\\mri\\hgg_valid_t1_10samples"]
        validation_directories = ["C:\\Users\\JanFiszer\\data\\mri\\hgg_valid_t1_10samples"]
    else:
        if len(sys.argv) > 1:
            data_dir = sys.argv[1]
            train_directories = [os.path.join(data_dir, "train")]
            validation_directories = [os.path.join(data_dir, "validation")]
            representative_test_dir = train_directories[0].split(os.path.sep)[-2]
        else:
            train_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/oasis/train",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/ucsf_150/train",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/lgg/train",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg_125/train",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hcp_mgh_masks/train",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hcp_wu_minn/train"]
            validation_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/oasis/validation",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/ucsf_150/validation",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/lgg/validation",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg_125/validation",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hcp_mgh_masks/validation",
                                 "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hcp_wu_minn/validation"]
            representative_test_dir = "all_data"


    translation_direction = config_train.TRANSLATION  # T1->T2
    train_dataset = MRIDatasetNumpySlices(train_directories, translation_direction=translation_direction, binarize=True)
    # train_dataset = MRIDatasetNumpySlices(train_directories, translation_direction=translation_direction, image_size=(176, 240))
    validation_dataset = MRIDatasetNumpySlices(validation_directories, translation_direction=translation_direction, binarize=True)

    print(f"Translation: {translation_direction[0].name}->{translation_direction[1].name}")


    if config_train.LOCAL:
        trainloader = DataLoader(train_dataset,
                                 batch_size=config_train.BATCH_SIZE)
        valloader = DataLoader(validation_dataset,
                               batch_size=config_train.BATCH_SIZE)
    else:
        num_workers = config_train.NUM_WORKERS
        print(f"Training with {num_workers} num_workers.")

        trainloader = DataLoader(train_dataset,
                                 batch_size=config_train.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=config_train.NUM_WORKERS,
                                 pin_memory=True)
        valloader = DataLoader(validation_dataset,
                               batch_size=config_train.BATCH_SIZE,
                               shuffle=True,
                               num_workers=config_train.NUM_WORKERS,
                               pin_memory=True)

    criterion = custom_metrics.loss_not_weighted_generalized_dice
    unet = UNet(criterion).to(config_train.DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)

    representative_test_dir = train_directories[0].split(os.path.sep)[-2]
    model_dir=f"{config_train.DATA_ROOT_DIR}/trained_models/model-{representative_test_dir}-{config_train.LOSS_TYPE.name}-ep{config_train.N_EPOCHS_CENTRALIZED}-{config_train.TRANSLATION[0].name}{config_train.TRANSLATION[1].name}-lr{config_train.LEARNING_RATE}-{config_train.now.date()}-{config_train.now.hour}h"
    fop.try_create_dir(model_dir)
    copy2("./configs/config_train.py", f"{model_dir}/config.py")

    if config_train.LOCAL:
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config_train.N_EPOCHS_CENTRALIZED,
                           filename="model.pth",
                           # model_dir=f"{config_train.DATA_ROOT_DIR}/trained_models/model-{representative_test_dir}-{config_train.LOSS_TYPE.name}-ep{config_train.N_EPOCHS_CENTRALIZED}-lr{config_train.LEARNING_RATE}-{config_train.NORMALIZATION.name}-{config_train.now.date()}-{config_train.now.hour}h",
                           history_filename="history.pkl")
    else:
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config_train.N_EPOCHS_CENTRALIZED,
                           filename="model.pth",
                           model_dir=model_dir,
                           history_filename="history.pkl",
                           plots_dir="predictions",
                           save_best_model=True)
