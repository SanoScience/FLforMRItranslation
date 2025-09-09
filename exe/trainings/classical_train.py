"""
Classical (centralized training) script for MRI translation.
Used as a baseline comparison for federated learning results.
"""

import sys
from shutil import copy2

import torch.optim as optim
from torch.utils.data import DataLoader

from src.ml.datasets import *
from src.ml.models import *


if __name__ == '__main__':
    # Set data directories based on environment
    if config_train.LOCAL:
        # Local testing paths
        train_directories = ["C:\\Users\\JanFiszer\\data\\mri\\fl-translation\\hgg_valid_t1_10samples"]
        validation_directories = ["C:\\Users\\JanFiszer\\data\\mri\\fl-translation\\hgg_valid_t1_10samples"]
    else:
        # Production paths from command line
        data_dir = sys.argv[1]
        train_directories = [os.path.join(data_dir, "train")]
        validation_directories = [os.path.join(data_dir, "validation")]
        # Extract dataset name for model directory naming
        representative_test_dir = train_directories[0].split(os.path.sep)[-2]

    # Configure dataset parameters
    translation_direction = config_train.TRANSLATION  # e.g., T1->T2 or FLAIR->T2
    dataset_kwargs = {
        "translation_direction": translation_direction, 
        "binarize": False,
        "normalize": True,
        "input_target_set_union": False
    }
    
    # Initialize datasets
    train_dataset = MRIDatasetNumpySlices(train_directories, **dataset_kwargs)
    validation_dataset = MRIDatasetNumpySlices(validation_directories, **dataset_kwargs)

    print(f"Translation: {translation_direction[0].name}->{translation_direction[1].name}")

    # Configure data loaders based on environment
    if config_train.LOCAL:
        # Simple loaders for local testing
        trainloader = DataLoader(train_dataset, batch_size=config_train.BATCH_SIZE)
        valloader = DataLoader(validation_dataset, batch_size=config_train.BATCH_SIZE)
    else:
        # Production loaders with parallel loading
        num_workers = config_train.NUM_WORKERS
        print(f"Training with {num_workers} num_workers.")

        trainloader = DataLoader(train_dataset,
                                 batch_size=config_train.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=config_train.NUM_WORKERS,
                                 pin_memory=True)  # Speed up GPU transfer
        valloader = DataLoader(validation_dataset,
                               batch_size=config_train.BATCH_SIZE,
                               shuffle=True,
                               num_workers=config_train.NUM_WORKERS,
                               pin_memory=True)

    # Initialize model components
    criterion = custom_metrics.DssimMseLoss()  # Combined SSIM and MSE loss
    unet = UNet(criterion).to(config_train.DEVICE)  # Move model to GPU if available
    mask = custom_metrics.init_mask(unet)
    optimizer = custom_metrics.MaskedAdam(unet.parameters(),  lr=config_train.LEARNING_RATE, mask=mask)

    # Create model directory with descriptive name
    representative_test_dir = train_directories[0].split(os.path.sep)[-2]
    model_dir = f"{config_train.DATA_ROOT_DIR}/trained_models/model-{representative_test_dir}-{config_train.LOSS_TYPE.name}-ep{config_train.N_EPOCHS_CENTRALIZED}-{config_train.TRANSLATION[0].name}{config_train.TRANSLATION[1].name}-lr{config_train.LEARNING_RATE}-{config_train.now.date()}-{config_train.now.hour}h"
    fop.try_create_dir(model_dir)
    # Save configuration for reproducibility
    copy2("./configs/config_train.py", f"{model_dir}/config.py")

    # Start training process
    if config_train.LOCAL:
        # Simple training for local testing
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config_train.N_EPOCHS_CENTRALIZED,
                           filename="model.pth",
                           model_dir=f"local-fl-training",
                           history_filename="history.pkl"
                           )
    else:
        # Full training with visualization and model checkpointing
        unet.perform_train(trainloader, optimizer,
                           validationloader=valloader,
                           epochs=config_train.N_EPOCHS_CENTRALIZED,
                           filename="model.pth",
                           model_dir=model_dir,
                           history_filename="history.pkl",
                           plots_dir="predictions",
                           save_best_model=True)  # Save model with best validation loss
