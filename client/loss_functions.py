from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from math import sqrt


def dssim_mse(predicted, targets):
    mse = MSELoss()
    ssim = StructuralSimilarityIndexMeasure()

    dssim = (1 - ssim(predicted, targets)) / 2

    return mse(predicted, targets) * dssim


