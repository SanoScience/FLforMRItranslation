import os
import sys

import torch

from common import models, datasets, utils, config_train
from client.utils import test

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")

if __name__ == '__main__':
    # test_dir = os.path.join(os.path.expanduser("~"), "data/hgg_transformed/validation")
    test_dir = sys.argv[1]
    model_path = sys.argv[2]
    BATCH_SIZE = int(sys.argv[3])

    testset = datasets.MRIDatasetNumpySlices([test_dir])
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    unet = models.UNet().to(config_train.DEVICE)
    if config_train.LOCAL:
        unet.load_state_dict(torch.load(os.path.join(model_path), map_location=torch.device('cpu')))
    else:
        try:
            unet.load_state_dict(torch.load(os.path.join(model_path)))
        except FileNotFoundError:
            FileNotFoundError(f"You are in {os.getcwd()} and there is no give path")

    images, targets = next(iter(testloader))

    images = images.to(config_train.DEVICE)
    predictions = unet(images)

    loss, ssim = test(unet, testloader)

    representative_test_dir = test_dir.split('/')[-2]
    filepath = os.path.join(model_path, f"test_{representative_test_dir}loss_-{loss:.4f}_ssim-{ssim:.4f}.jpg")
    utils.plot_predicted_batch(images.cpu(), targets.cpu(), predictions.cpu(),
                               title=f"loss: {loss} ssim: {ssim}",
                               show=False,
                               filepath=filepath)

    print("Plot saved to :", filepath)
    print("\nEvaluation ended.\n")
