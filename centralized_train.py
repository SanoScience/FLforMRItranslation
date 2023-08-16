import os
import torch.optim as optim
from torch.nn import MSELoss

from src import loss_functions
from src.datasets import *
from src.models import *

# for ares when in the home directory
if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")

    train_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg/train"]
                        #  "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/lgg/train",
                        #  "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hcp_mgh_masks/train",
                        #  "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hcp_wu_minn"]
    validation_directories = ["/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg/validation"]
                            #   "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/lgg/validation",
                            #   "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hcp_mgh_masks/validation",
                            #   "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hcp_wu_minn/validation"]
else:
    train_directories = "C:\\Users\\JanFiszer\\data\\hgg_transformed\\validation"
    validation_directories = "C:\\Users\\JanFiszer\\data\\hgg_transformed\\validation"
    # ROOT_DIR_TRAIN = os.path.join(os.path.expanduser("~"), "data/HGG")

if __name__ == '__main__':
    train_dataset = MRIDatasetNumpySlices(train_directories)
    validation_dataset = MRIDatasetNumpySlices(validation_directories)

    num_workers = config_train.NUM_WORKERS
    print(f"num_workers: {num_workers}")

    trainloader = DataLoader(train_dataset,
                             batch_size=config_train.BATCH_SIZE,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True)
    valloader = DataLoader(validation_dataset,
                           batch_size=config_train.BATCH_SIZE,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True
                           )

    # train_dataset = MRIDatasetNumpySlices([ROOT_DIR_TRAIN])
    # validation_dataset = MRIDatasetNumpySlices([ROOT_DIR_VAL])
    # trainloader = DataLoader(train_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)
    # valloader = DataLoader(validation_dataset, batch_size=config_train.BATCH_SIZE, shuffle=True)

    unet = UNet().to(config_train.DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=config_train.LEARNING_RATE)
    criterion = MSELoss()
    model_filename = f"model_real_without_norm.pth"

    print()
    print(unet)
    print()

    unet.perform_train(trainloader,
                       optimizer,
                       criterion,
                       validationloader=valloader,
                       epochs=config_train.N_EPOCHS_CENTRALIZED,
                       filename=model_filename,
                       plots_dir="predictions", 
                       history_filename="history.pkl")
