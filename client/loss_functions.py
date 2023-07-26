import torch
from common import config_train
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure


def dssim_mse(predicted, targets):
    mse = MSELoss()
    ssim = StructuralSimilarityIndexMeasure().to(config_train.DEVICE)

    dssim = (1 - ssim(predicted, targets)) / 2

    return mse(predicted, targets) * dssim


