from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from configs import config_train
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric
from scipy import signal


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
        return (sum(self.ssim_list) / self.batch_size).clone().detach()


class QILV(Metric):
    def __init__(self, use_mask=True, **kwargs):
        super().__init__(**kwargs)
        self.add_state("qilv_list", default=[], dist_reduce_fx="cat")
        self.add_state("batch_size", default=torch.tensor(0), dist_reduce_fx="sum")
        self.use_mask = use_mask

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        for pred, target in zip(preds, targets):
            self.qilv_list.append(self._compute_QILV(pred, target))

        self.batch_size += preds.shape[0]

    def compute(self):
        return (sum(self.qilv_list) / self.batch_size).clone().detach()


    @staticmethod
    def nanstd(x):
        mask = torch.isnan(x)
        n = torch.sum(~mask)
        if n > 0:
            mean = torch.sum(x[~mask]) / n
            variance = torch.sum((x[~mask] - mean) ** 2) / n
            return torch.sqrt(variance)
        else:
            return torch.tensor(float('nan'), dtype=x.dtype)

    def _compute_QILV(self, pred, target, window=0):
        if window == 0:
            window = torch.tensor(signal.windows.gaussian(11, std=1.5), dtype=torch.float32)

            # Outer product (make 2D window)
            window = torch.outer(window, window)

        # Normalize window
        window = window / torch.sum(window)

        unsqueezed_window = window.unsqueeze(0).unsqueeze(0)
        unsqueezed_target = target.unsqueeze(0)
        unsqueezed_pred = pred.unsqueeze(0)

        # Local means
        M1 = F.conv2d(unsqueezed_target, unsqueezed_window, padding=5)
        M2 = F.conv2d(unsqueezed_pred, unsqueezed_window, padding=5)

        # Local variances
        V1 = F.conv2d(unsqueezed_target ** 2, unsqueezed_window, padding=5) - M1 ** 2
        V2 = F.conv2d(unsqueezed_pred ** 2, unsqueezed_window, padding=5) - M2 ** 2

        if self.use_mask:
            mask = target == 0

        else:
            mask = torch.ones_like(target)

        V1 = torch.where(mask, V1, torch.tensor(np.nan, dtype=torch.float32))
        V2 = torch.where(mask, V2, torch.tensor(np.nan, dtype=torch.float32))

        m1 = torch.nanmean(V1)
        m2 = torch.nanmean(V2)
        s1 = self.nanstd(V1)
        s2 = self.nanstd(V2)

        s12 = torch.nanmean((V1 - m1) * (V2 - m2))

        # QILV index
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        ind1 = ((2 * m1 * m2 + C1) / (m1 ** 2 + m2 ** 2 + C1))
        ind2 = (2 * s1 * s2 + C2) / (s1 ** 2 + s2 ** 2 + C2)
        ind3 = (s12 + C2 / 2) / (s1 * s2 + C2 / 2)
        ind = ind1 * ind2 * ind3

        return ind

def QILV_nontorch(image1, image2, mask=None, window=0):
    if (window == 0):
        window = signal.gaussian(11, std=1.5)

        # outer product (make 2D window)
        window = np.outer(window, window)

    # normalize window
    window = window / np.sum(window)

    # local means
    M1 = signal.convolve2d(image1, window, mode='same')
    M2 = signal.convolve2d(image2, window, mode='same')

    # local variances
    V1 = signal.convolve2d((image1 ** 2), window, mode='same') - M1 ** 2
    V2 = signal.convolve2d((image2 ** 2), window, mode='same') - M2 ** 2

    # global statistics
    if (mask != None).all():
        mask = np.ones(image1.shape)

    mask = np.invert(mask > 0)

    V1 = np.ma.array(V1, mask=mask)
    V2 = np.ma.array(V2, mask=mask)

    m1 = V1.mean()
    m2 = V2.mean()
    s1 = V1.std()
    s2 = V2.std()

    print(m1, m2, s1, s2)

    s12 = np.mean((V1 - m1) * (V2 - m2))

    # QILV index
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ind1 = ((2 * m1 * m2 + C1) / (m1 ** 2 + m2 ** 2 + C1))
    ind2 = (2 * s1 * s2 + C2) / (s1 ** 2 + s2 ** 2 + C2)
    ind3 = (s12 + C2 / 2) / (s1 * s2 + C2 / 2)
    ind = ind1 * ind2 * ind3

    return ind


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
