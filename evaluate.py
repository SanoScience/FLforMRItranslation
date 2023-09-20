import os
import sys
import pickle
import torch

from configs import config_train
from src import datasets, models, loss_functions, visualization
from torch.utils.data import DataLoader

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")

if __name__ == '__main__':
    print(sys.argv)
    test_dir = sys.argv[1]
    model_path = sys.argv[2]
    BATCH_SIZE = int(sys.argv[3])

    testset = datasets.MRIDatasetNumpySlices([test_dir])
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    # unet = models.UNet().to(config_train.DEVICE)
    criterion = loss_functions.DssimMse()
    unet = models.UNet(criterion).to(config_train.DEVICE)

    try:
        unet.load_state_dict(torch.load(os.path.join(model_path)))
    except FileNotFoundError:
        FileNotFoundError(f"You are in {os.getcwd()} and there is no give path")

    images, targets = next(iter(testloader))

    images = images.to(config_train.DEVICE)
    predictions = unet(images)

    metrics = unet.evaluate(testloader)
    representative_test_dir = test_dir.split('/')[-2]
    model_dir = '/'.join(e for e in model_path.split('/')[:-1])
  
    filepath = os.path.join(model_dir, f"test_{representative_test_dir}_loss_{metrics['loss']:.2f}.pkl")

    with open(filepath, "wb") as file:
        pickle.dump(metrics, file)
    print(f"Saved to : {filepath}")
