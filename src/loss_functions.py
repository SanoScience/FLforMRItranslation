from typing import Dict, List

import torch
from configs import config_train
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure


class DssimMse:
    def __init__(self, sqrt=False):
        self.sqrt = sqrt

    def __call__(self, predicted, targets):
        mse = MSELoss()
        ssim = StructuralSimilarityIndexMeasure().to(config_train.DEVICE)

        dssim = (1 - ssim(predicted, targets)) / 2

        if self.sqrt:
            result = torch.sqrt_(mse(predicted, targets)) + dssim
        else:
            result = mse(predicted, targets) + dssim
        return result

    def __repr__(self):
        if self.sqrt:
            representation = f"DSSIM RootMSE loss function"
        else:
            representation = f"DSSIM MSE loss function"
        return representation

    def __str__(self):
        if self.sqrt:
            representation = f"DSSIM RootMSE"
        else:
            representation = f"DSSIM MSE"
        return representation


class LossWithProximalTerm:
    def __init__(self, proximal_mu, base_loss_fn):
        self.proximal_mu = proximal_mu
        self.base_loss_fn = base_loss_fn

    def __call__(self, predicted, targets, local_params, global_params):
        proximal_term = 0.0

        for local_weights, global_weights in zip(local_params, global_params):
            proximal_term += (local_weights - global_weights).norm(2) ** 2

        return self.base_loss_fn(predicted, targets) + (self.proximal_mu/2) * proximal_term

    def __repr__(self):
        return f"ProxLoss with mu={self.proximal_mu} base function={self.base_loss_fn}"

    def __str__(self):
        return f"ProxLoss (mu={self.proximal_mu} base_fn={self.base_loss_fn})"


def loss_for_config():
    if config_train.LOSS_TYPE == config_train.LossFunctions.MSE_DSSIM:
        return DssimMse()
    elif config_train.LOSS_TYPE == config_train.LossFunctions.PROX:
        return LossWithProximalTerm(config_train.PROXIMAL_MU, DssimMse())
    elif config_train.LOSS_TYPE == config_train.LossFunctions.RMSE_DDSIM:
        return DssimMse(sqrt=True)
    else:  # config_train.LOSS_TYPE == config_train.LossFunctions.MSE:
        return MSELoss()


def metrics_to_str(metrics: Dict[str, List[float]], starting_symbol="\t"):
    metrics_epoch_str = starting_symbol
    for metric_name, epoch_value in metrics.items():
        metrics_epoch_str += f"{metric_name}: {epoch_value:.3f}\t"

    return metrics_epoch_str
