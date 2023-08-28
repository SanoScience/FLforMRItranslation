from typing import Dict, List

import numpy as np
import torch
from configs import config_train
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric


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


class ImprovedSSIM(Metric):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.add_state("summed_ssim", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("batch_size", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(config_train.DEVICE)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds, targets = self._input_format()
        assert preds.shape == targets.shape

        for pred, target in zip(preds, targets):
            image = target.detach()[0]

            looking_for_first = True
            first_index_row = -1
            last_index_row = -1

            for index, row in enumerate(image):
                if looking_for_first and torch.sum(row) > 0:
                    first_index_row = index
                    looking_for_first = False
                if not looking_for_first and torch.sum(row) == 0:
                    last_index_row = index
                    looking_for_first = True
                    break

            first_index_col = -1
            last_index_col = -1

            for index, col in enumerate(torch.t(image)):
                if looking_for_first and sum(col) > 0:
                    first_index_col = index
                    looking_for_first = False
                if not looking_for_first and sum(col) == 0:
                    last_index_col = index
                    break

            trimmed_targets = torch.tensor(image[first_index_row:last_index_row, first_index_col:last_index_col])
            trimmed_predicted = torch.tensor(pred[0, first_index_row:last_index_row, first_index_col:last_index_col])

            # expanding dimensions to fit BxCxHxW format
            self.summed_ssim += self.ssim(trimmed_predicted[None, None, :, :], trimmed_targets[None, None, :, :])

        self.batch_size += preds.shape[0]

    def compute(self):
        return self.summed_ssim / self.batch_size

    # def __call__(self, predicted, targets):
    #     for image in targets:
    #         image_np = image.detach().numpy()[0]
    #
    #         looking_for_first = True
    #         first_index_row = -1
    #         last_index_row = -1
    #
    #         for index, row in enumerate(image_np):
    #             if looking_for_first and sum(row) > 0:
    #                 first_index_row = index
    #                 looking_for_first = False
    #             if not looking_for_first and sum(row) == 0 :
    #                 last_index_row = index
    #                 looking_for_first = True
    #                 break
    #
    #         first_index_col = -1
    #         last_index_col = -1
    #
    #         for index, col in enumerate(image_np.transpose()):
    #             if looking_for_first and sum(col) > 0:
    #                 first_index_col = index
    #                 looking_for_first = False
    #             if not looking_for_first and sum(col) == 0:
    #                 last_index_col = index
    #                 break
    #
    #         trimmed_targets = torch.tensor(image_np[first_index_row:last_index_row, first_index_col:last_index_col])
    #         trimmed_predicted = torch.tensor(predicted[first_index_row:last_index_row, first_index_col:last_index_col])
    #
    #     return self.ssim(trimmed_predicted, trimmed_targets)

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
