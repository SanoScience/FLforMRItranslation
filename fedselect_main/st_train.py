from torch.utils.data import DataLoader

from fedselect_main.fedselect import train_personalized
from src.ml.datasets import MRIDatasetNumpySlices
from src.ml.models import UNet
from configs import config_train, enums
from src.ml import custom_metrics, datasets
from fedselect_main.lottery_ticket import init_mask_zeros, delta_update, init_mask
from fedselect_main.utils.options import lth_args_parser

import sys
import os

if __name__ == "__main__":
    args = lth_args_parser()

    # Production paths from command line
    data_dir = args.data_dir
    train_directories = [os.path.join(data_dir, "train")]
    validation_directories = [os.path.join(data_dir, "validation")]

    # Configure dataset parameters
    translation_direction = config_train.TRANSLATION  # e.g., T1->T2 or FLAIR->T2
    dataset_kwargs = {
        "translation_direction": translation_direction,
        "binarize": False,  # Keep original intensity values
        "normalize": True,
        "input_target_set_union": False  # Use all available data
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

    train_personalized(unet, trainloader, init_mask(unet), args)

    unet.evaluate(valloader, plots_path=os.path.join(config_train.ROOT_DIR, "trained_models", "st_model_fedselect_optimizer"), plot_filename="after_training.jpg", plot_metrics_distribution=True)

