from typing import Dict, List

import numpy as np
import torch
from configs import config_train
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric


class DssimMse:
    def __init__(self, sqrt=False, zoomed_ssim=False):
        self.zoomed_ssim = zoomed_ssim
        self.sqrt = sqrt

    def __call__(self, predicted, targets):
        mse = MSELoss()
        if self.zoomed_ssim:
            ssim = ZoomedSSIM()
        else:
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
        representation = "DSSIM"

        if self.sqrt:
            representation += " RootMSE"
        else:
            representation += f" MSE"

        if self.zoomed_ssim:
            representation += " (zoomed_ssim=True)"
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


class ZoomedSSIM(Metric):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.add_state("ssim_list", default=[], dist_reduce_fx="cat")
        self.add_state("batch_size", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(config_train.DEVICE)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):

        # preds, targets = self._input_format()
        assert preds.shape == targets.shape

        for pred, target in zip(preds, targets):
            image = target.detach()[0]

            non_zeros = torch.nonzero(image)
            first_index_row = int(non_zeros[0][0])
            last_index_row = int(non_zeros[-1][0])

            non_zeros_T = torch.nonzero(torch.t(image))
            first_index_col = int(non_zeros_T[0][0])
            last_index_col = int(non_zeros_T[-1][0])

            trimmed_targets = image.detach().clone()[first_index_row:last_index_row, first_index_col:last_index_col]
            trimmed_predicted = pred.detach().clone()[0, first_index_row:last_index_row, first_index_col:last_index_col]

            # expanding dimensions to fit BxCxHxW format
            ssim_value = self.ssim(trimmed_predicted[None, None, :, :], trimmed_targets[None, None, :, :])
            self.ssim_list.append(ssim_value)

        self.batch_size += preds.shape[0]


    def compute(self):
        return torch.tensor(sum(self.ssim_list) / self.batch_size)


def loss_from_config():
    if config_train.LOSS_TYPE == config_train.LossFunctions.MSE_DSSIM:
        return DssimMse()
    elif config_train.LOSS_TYPE == config_train.LossFunctions.PROX:
        return LossWithProximalTerm(config_train.PROXIMAL_MU, DssimMse())
    elif config_train.LOSS_TYPE == config_train.LossFunctions.RMSE_DDSIM:
        return DssimMse(sqrt=True)
    elif config_train.LOSS_TYPE == config_train.LossFunctions.MSE_ZOOMED_DSSIM:
        return DssimMse(zoomed_ssim=True)
    else:  # config_train.LOSS_TYPE == config_train.LossFunctions.MSE:
        return MSELoss()


def metrics_to_str(metrics: Dict[str, List[float]], starting_symbol=""):
    metrics_epoch_str = starting_symbol
    for metric_name, epoch_value in metrics.items():
        metrics_epoch_str += f"{metric_name}: {epoch_value:.3f}\t"

    return metrics_epoch_str
