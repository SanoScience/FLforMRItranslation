import os
import sys
import torch.optim as optim
from torch.nn import MSELoss

from src import loss_functions
from src.datasets import *
from src.models import *


if __name__ == '__main__':

    if config_train.LOCAL:
        train_directories = ["C:\\Users\\JanFiszer\\data\\mri\\mgh_on_hpc\\train"]
        validation_directories = ["C:\\Users\\JanFiszer\\data\\mri\\mega_small_hgg\\validation"]
    else:
        if len(sys.argv) > 1:
            data_dir = sys.argv[1]
            train_directories = [os.path.join(data_dir, "train")]
            validation_directories = [os.path.join(data_dir, "validation")]
        else:
            train_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/oasis_125/train"]
            validation_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/oasis_125/validation"]

    translation_direction = (enums.ImageModality.T1, enums.ImageModality.T2)  # T1->T2
    train_dataset = MRIDatasetNumpySlices(train_directories, translation_direction=translation_direction)
    validation_dataset = MRIDatasetNumpySlices(validation_directories, translation_direction=translation_direction)

    print(f"Translation: {translation_direction[0].name}->{translation_direction[1].name}")

    representative_test_dir = train_directories[0].split(os.path.sep)[-2]

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
                               num_workers=config_train.NUM_WORKERS,
                               pin_memory=True)

    criterion = loss_functions.DssimMse()
    unet = UNet(criterion).to(config_train.DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)

    if config_train.LOCAL:
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config_train.N_EPOCHS_CENTRALIZED,
                           filename="model.pth",
                           # model_dir=f"{config_train.DATA_ROOT_DIR}/trained_models/model-{representative_test_dir}-{config_train.LOSS_TYPE.name}-ep{config_train.N_EPOCHS_CENTRALIZED}-lr{config_train.LEARNING_RATE}-{config_train.NORMALIZATION.name}-{config_train.now.date()}-{config_train.now.hour}h",
                           history_filename="history.pkl")
    else:
        representative_test_dir = train_directories[0].split('/')[-2]
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config_train.N_EPOCHS_CENTRALIZED,
                           filename="model.pth",
                           model_dir=f"{config_train.DATA_ROOT_DIR}/trained_models/model-{representative_test_dir}-{config_train.LOSS_TYPE.name}-ep{config_train.N_EPOCHS_CENTRALIZED}-lr{config_train.LEARNING_RATE}-{config_train.NORMALIZATION.name}-{config_train.now.date()}-{config_train.now.hour}h",
                           history_filename="history.pkl",
                           plots_dir="predictions",
                           save_best_model=True)
