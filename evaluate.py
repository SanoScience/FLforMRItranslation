import os
import sys

import torch

from common import models, datasets, utils, config_train
from client.utils import test

if __name__ == '__main__':
    # test_dir = os.path.join(os.path.expanduser("~"), "data/hgg_transformed/validation")
    test_dir = sys.argv[1]
    BATCH_SIZE = int(sys.argv[2])

    testset = datasets.MRIDatasetNumpySlices([test_dir])
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    unet = models.UNet()
    if config_train.LOCAL:
        unet.load_state_dict(torch.load("trained_models/client/model-lr0.001-ep20-2023-07-26-12_36/model.pth",  map_location=torch.device('cpu')))
    else:
        unet.load_state_dict(torch.load("trained_models/client/model-lr0.001-ep20-2023-07-26-12_36/model.pth"))

    images, targets = next(iter(testloader))
    predictions = unet(images)

    loss, ssim = test(unet, testloader)

    filepath = os.path.join(config_train.TRAINED_MODEL_CLIENT_DIR, f"test_loss-{loss}_ssim-{ssim}.jpg")
    utils.plot_predicted_batch(images, targets, predictions,
                               title=f"loss: {loss} ssim: {ssim}",
                               show=False,
                               filepath=filepath)
