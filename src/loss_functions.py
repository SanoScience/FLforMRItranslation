import torch
from configs import config_train
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure


def dssim_mse(predicted, targets):
    mse = MSELoss()
    ssim = StructuralSimilarityIndexMeasure().to(config_train.DEVICE)

    dssim = (1 - ssim(predicted, targets)) / 2

    return mse(predicted, targets) * dssim


class LossWithProximalTerm:
    def __init__(self, proximal_mu, base_loss_fn):
        self.proximal_mu = proximal_mu
        self.base_loss_fn = base_loss_fn

    def __call__(self, predicted, targets, local_params, global_params, *args, **kwargs):
        proximal_term = 0.0

        for local_weights, global_weights in zip(local_params, global_params):
            # TODO: deal with strange .data
            proximal_term += (local_weights - global_weights).norm(2)

        return self.base_loss_fn(predicted, targets) + (self.proximal_mu/2) * proximal_term

    def __repr__(self):
        return f"ProxLoss with mu={self.proximal_mu} base function={self.base_loss_fn}"


