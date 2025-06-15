from typing import Dict, List, Any

import numpy as np
from configs import config_train
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric
from scipy import signal

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

# for own torch implementation
from torchmetrics.functional.image.helper import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torchmetrics.functional.image.ssim import _multiscale_ssim_update, _ssim_check_inputs
from torchmetrics.utilities.data import dim_zero_cat

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide


class DssimMse:
    def __init__(self, sqrt=False, zoomed_ssim=False):
        self.sqrt = sqrt
        self.mse = MSELoss()
        self.zoomed_ssim = zoomed_ssim

        if zoomed_ssim:
            self.ssim = ZoomedSSIM()
        else:
            self.ssim = StructuralSimilarityIndexMeasure().to(config_train.DEVICE)

    def __call__(self, predicted, targets):
        dssim = (1 - self.ssim(predicted, targets)) / 2

        if self.sqrt:
            result = torch.sqrt_(self.mse(predicted, targets)) + dssim
        else:
            result = self.mse(predicted, targets) + dssim
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

        return self.base_loss_fn(predicted, targets) + (self.proximal_mu / 2) * proximal_term

    def __repr__(self):
        return f"ProxLoss with mu={self.proximal_mu} base function={self.base_loss_fn}"

    def __str__(self):
        return f"ProxLoss (mu={self.proximal_mu} base_fn={self.base_loss_fn})"


class ZoomedSSIM(Metric):
    def __init__(self, data_range=1.0, margin=0, min_mask_size=10):
        super().__init__()
        self.add_state("ssim_list", default=[], dist_reduce_fx="cat")
        self.add_state("batch_size", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(config_train.DEVICE)

        self.margin = margin
        self.min_mask_size=min_mask_size

    def update(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None):
        # preds, targets = self._input_format()
        assert preds.shape == targets.shape

        for pred, target, mask in zip(preds, targets, masks):
            image = target.detach()[0]

            if mask is None:  # default the brain mask which can be easily computed
                non_zeros = torch.nonzero(image)
                non_zeros_T = torch.nonzero(torch.t(image))
            else:
                # print(f"Mask characteriscs are {torch.unique(mask, return_counts=True)}")
                non_zeros = torch.nonzero(mask)
                non_zeros_T = torch.nonzero(torch.t(mask))

                # in case in the picture is no mask 
                # then the evaluation is skipped
                # and the batch size is decreased 
                mask_size = torch.sum(mask)
                if mask_size < self.min_mask_size:
                    print(f"Skipped due to too small mask size {mask_size}")
                    self.batch_size -= 1
                    continue

            first_index_row = int(non_zeros[0][0]) - self.margin
            last_index_row = int(non_zeros[-1][0]) + self.margin

            first_index_col = int(non_zeros_T[0][0]) - self.margin
            last_index_col = int(non_zeros_T[-1][0]) + self.margin

            # print(f"taken indices:" , first_index_col, last_index_col, first_index_row, last_index_row)

            trimmed_targets = image.detach().clone()[first_index_row:last_index_row, first_index_col:last_index_col]
            trimmed_predicted = pred.detach().clone()[0, first_index_row:last_index_row, first_index_col:last_index_col]

            # expanding dimensions to fit BxCxHxW format
            ssim_value = self.ssim(trimmed_predicted[None, None, :, :], trimmed_targets[None, None, :, :])
            self.ssim_list.append(ssim_value)

        self.batch_size += preds.shape[0]

    def compute(self):
        return (sum(self.ssim_list) / self.batch_size).clone().detach()


class MaskedSSIM(Metric):
    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
            self,
            gaussian_kernel: bool = True,
            sigma: Union[float, Sequence[float]] = 1.5,
            kernel_size: Union[int, Sequence[int]] = 11,
            reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
            data_range: Optional[Union[float, Tuple[float, float]]] = None,
            k1: float = 0.01,
            k2: float = 0.03,
            return_full_image: bool = False,
            return_contrast_sensitivity: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        valid_reduction = ("elementwise_mean", "sum", "none", None)
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")

        if reduction in ("elementwise_mean", "sum"):
            self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("similarity", default=[], dist_reduce_fx="cat")

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        if return_contrast_sensitivity or return_full_image:
            self.add_state("image_return", default=[], dist_reduce_fx="cat")

        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.return_full_image = return_full_image
        self.return_contrast_sensitivity = return_contrast_sensitivity

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _ssim_check_inputs(preds, target)
        similarity_pack = self._ssim_update(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.data_range,
            self.k1,
            self.k2,
            self.return_full_image,
            self.return_contrast_sensitivity,
        )

        if isinstance(similarity_pack, tuple):
            similarity, image = similarity_pack
        else:
            similarity = similarity_pack

        if self.return_contrast_sensitivity or self.return_full_image:
            self.image_return.append(image)

        if self.reduction in ("elementwise_mean", "sum"):
            self.similarity += similarity.sum()
            self.total += preds.shape[0]
        else:
            self.similarity.append(similarity)

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute SSIM over state."""
        if self.reduction == "elementwise_mean":
            similarity = self.similarity / self.total
        elif self.reduction == "sum":
            similarity = self.similarity
        else:
            similarity = dim_zero_cat(self.similarity)

        if self.return_contrast_sensitivity or self.return_full_image:
            image_return = dim_zero_cat(self.image_return)
            return similarity, image_return

        return similarity

    @staticmethod
    def _ssim_update(
            preds: Tensor,
            target: Tensor,
            gaussian_kernel: bool = True,
            sigma: Union[float, Sequence[float]] = 1.5,
            kernel_size: Union[int, Sequence[int]] = 11,
            data_range: Optional[Union[float, Tuple[float, float]]] = None,
            k1: float = 0.01,
            k2: float = 0.03,
            return_full_image: bool = False,
            return_contrast_sensitivity: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute Structual Similarity Index Measure.

        Args:
            preds: estimated image
            target: ground truth image
            gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
            sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
                Ignored if a uniform kernel is used
            kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
                Ignored if a Gaussian kernel is used
            data_range: Range of the image. If ``None``, it is determined from the image (max - min)
            k1: Parameter of SSIM.
            k2: Parameter of SSIM.
            return_full_image: If true, the full ``ssim`` image is returned as a second argument.
                Mutually exlusive with ``return_contrast_sensitivity``
            return_contrast_sensitivity: If true, the contrast term is returned as a second argument.
                The luminance term can be obtained with luminance=ssim/contrast
                Mutually exclusive with ``return_full_image``
        """
        is_3d = preds.ndim == 5

        if not isinstance(kernel_size, Sequence):
            kernel_size = 3 * [kernel_size] if is_3d else 2 * [kernel_size]
        if not isinstance(sigma, Sequence):
            sigma = 3 * [sigma] if is_3d else 2 * [sigma]

        if len(kernel_size) != len(target.shape) - 2:
            raise ValueError(
                f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
                f" which is: {len(target.shape)}"
            )
        if len(kernel_size) not in (2, 3):
            raise ValueError(
                f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
            )
        if len(sigma) != len(target.shape) - 2:
            raise ValueError(
                f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
                f" which is: {len(target.shape)}"
            )
        if len(sigma) not in (2, 3):
            raise ValueError(
                f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
            )

        if return_full_image and return_contrast_sensitivity:
            raise ValueError("Arguments `return_full_image` and `return_contrast_sensitivity` are mutually exclusive.")

        if any(x % 2 == 0 or x <= 0 for x in kernel_size):
            raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in sigma):
            raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

        if data_range is None:
            data_range = max(preds.max() - preds.min(), target.max() - target.min())
        elif isinstance(data_range, tuple):
            preds = torch.clamp(preds, min=data_range[0], max=data_range[1])
            target = torch.clamp(target, min=data_range[0], max=data_range[1])
            data_range = data_range[1] - data_range[0]

        c1 = pow(k1 * data_range, 2)
        c2 = pow(k2 * data_range, 2)
        device = preds.device

        mask = target > 0

        channel = preds.size(1)
        dtype = preds.dtype
        gauss_kernel_size = [int(3.5 * s + 0.5) * 2 + 1 for s in sigma]

        pad_h = (gauss_kernel_size[0] - 1) // 2
        pad_w = (gauss_kernel_size[1] - 1) // 2

        if is_3d:
            pad_d = (gauss_kernel_size[2] - 1) // 2
            preds = _reflection_pad_3d(preds, pad_d, pad_w, pad_h)
            target = _reflection_pad_3d(target, pad_d, pad_w, pad_h)
            if gaussian_kernel:
                kernel = _gaussian_kernel_3d(channel, gauss_kernel_size, sigma, dtype, device)
        else:
            preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
            target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
            if gaussian_kernel:
                kernel = _gaussian_kernel_2d(channel, gauss_kernel_size, sigma, dtype, device)

        if not gaussian_kernel:
            kernel = torch.ones((channel, 1, *kernel_size), dtype=dtype, device=device) / torch.prod(
                torch.tensor(kernel_size, dtype=dtype, device=device)
            )

        input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)

        outputs = F.conv3d(input_list, kernel, groups=channel) if is_3d else F.conv2d(input_list, kernel,
                                                                                      groups=channel)

        output_list = outputs.split(preds.shape[0])

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        upper = 2 * sigma_pred_target.to(dtype) + c2
        lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c2

        ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)

        ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w]

        if return_contrast_sensitivity:
            contrast_sensitivity = upper / lower
            if is_3d:
                contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d]
            else:
                contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w]
            return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), contrast_sensitivity.reshape(
                contrast_sensitivity.shape[0], -1
            ).mean(-1)

        ssim_idx_masked = ssim_idx * mask[..., pad_h:-pad_h, pad_w:-pad_w]

        return ssim_idx_masked.reshape(ssim_idx_masked.shape[0], -1).sum(-1) / mask.reshape(ssim_idx_masked.shape[0],
                                                                                            -1).sum(-1)


class MaskedMSE:
    def __call__(self, predicted, targets):
        mask = targets > 0
        return torch.sum(((predicted - targets) * mask) ** 2.0) / torch.sum(mask)


class RelativeError:
    def __call__(self, predicted, targets, *args: Any, **kwds: Any) -> Any:
        mask = targets > 0
        result = torch.nansum((abs(predicted - targets) / targets) * mask) / torch.sum(mask)
        return result


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