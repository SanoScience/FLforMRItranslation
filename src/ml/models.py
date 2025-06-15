import logging
from os import path
import pickle

import matplotlib.pyplot as plt
import wandb
import time
from typing import Callable, Optional, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from configs import config_train, creds
from src.ml import custom_metrics
from src.utils import files_operations as fop, visualization

device = config_train.DEVICE
batch_print_freq = config_train.BATCH_PRINT_FREQ
ssim = StructuralSimilarityIndexMeasure().to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
mse = nn.MSELoss()
masked_mse = custom_metrics.MaskedMSE()
relative_error = custom_metrics.RelativeError()
masked_ssim = custom_metrics.MaskedSSIM().to(device)
zoomed_ssim = custom_metrics.ZoomedSSIM(margin=5)



class UNet(nn.Module):
    """
    UNet model for MRI translation with configurable normalization.
    
    Attributes:
        criterion: Loss function for training
        bilinear: Whether to use bilinear upsampling
        available_metrics: Dictionary of evaluation metrics
        normalization: Normalization type to be used in the model
    """
    def __init__(self, criterion: Optional[Callable] = None, 
                 bilinear: bool = False, 
                 normalization: enums.NormalizationType = config_train.NORMALIZATION) -> None:
        super(UNet, self).__init__()

        self.criterion = criterion
        self.bilinear = bilinear

        self.available_metrics = {"loss": self.criterion,
                                  "ssim": ssim,
                                  "pnsr": psnr,
                                  "mse": mse,
                                  "masked_mse": masked_mse,
                                  "masked_ssim": masked_ssim,
                                  "relative_error": relative_error,
                                  "zoomed_ssim": zoomed_ssim}

        self.inc = (DoubleConv(1, 64, normalization))
        self.down1 = (Down(64, 128, normalization))
        self.down2 = (Down(128, 256, normalization))
        self.down3 = (Down(256, 512, normalization))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, normalization))
        self.up1 = (Up(1024, 512 // factor, normalization, bilinear))
        self.up2 = (Up(512, 256 // factor, normalization, bilinear))
        self.up3 = (Up(256, 128 // factor, normalization, bilinear))
        self.up4 = (Up(128, 64, normalization, bilinear))
        self.outc = (OutConv(64, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def save(self, dir_name: str, filename=None):
        """
        Saves the model to a given directory. Allows to change the name of the file, by default it is "model".
        """
        if filename is None:
            filename = "model"

        if not isinstance(dir_name, str):
            raise TypeError(f"Given directory name {dir_name} has wrong type: {type(dir_name)}.")

        fop.try_create_dir(dir_name)

        if not filename.endswith(".pth"):
            filename += ".pth"

        filepath = f"{dir_name}/{filename}"
        torch.save(self.state_dict(), filepath)

        print("Model saved to: ", filepath)

    def _train_one_epoch(self, trainloader: DataLoader, 
                        optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Method used by perform_train(). Does one iteration of training.
        """
        self.train()

        metrics = {metric_name: self.available_metrics[metric_name] for metric_name in config_train.METRICS}

        epoch_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}
        total_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}

        n_batches = len(trainloader)

        start = time.time()
        n_train_steps = 0

        use_prox_loss = isinstance(self.criterion, custom_metrics.LossWithProximalTerm)

        if n_batches < config_train.BATCH_PRINT_FREQ:
            batch_print_frequency = n_batches - 2  # tbh not sure if this -2 is needed
        else:
            batch_print_frequency = config_train.BATCH_PRINT_FREQ

        if use_prox_loss:
            global_params = [val.detach().clone() for val in self.parameters()]

        for index, data in enumerate(trainloader):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = self(images)

            if use_prox_loss:
                loss = self.criterion(predictions, targets, self.parameters(), global_params)
            else:
                loss = self.criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            for metric_name, metric_object in metrics.items():
                if metric_name == "loss":
                    metric_value = loss
                else:
                    metric_value = metric_object(predictions, targets)
                total_metrics[metric_name] += metric_value.item()
                epoch_metrics[metric_name] += metric_value.item()

            n_train_steps += 1

            if index % batch_print_frequency == batch_print_frequency - 1:
                divided_batch_metrics = {metric_name: total_value / batch_print_frequency for metric_name, total_value
                                         in total_metrics.items()}
                metrics_str = custom_metrics.metrics_to_str(divided_batch_metrics, starting_symbol="\t")
                print(f'\t\tbatch {(index + 1)} out of {n_batches}\t\t{metrics_str}')

                total_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}

        averaged_epoch_metrics = {metric_name: metric_value / n_train_steps for metric_name, metric_value in
                                  epoch_metrics.items()}
        metrics_epoch_str = custom_metrics.metrics_to_str(averaged_epoch_metrics, starting_symbol="")

        print(f"\n\tTime exceeded: {time.time() - start:.1f}")
        print(f"\tEpoch metrics: {metrics_epoch_str}")

        return averaged_epoch_metrics

    def perform_train(self,
                     trainloader: DataLoader,
                     optimizer: torch.optim.Optimizer,  
                     epochs: int,
                     validationloader: Optional[DataLoader] = None,
                     model_dir: str = config_train.TRAINED_MODEL_SERVER_DIR,
                     filename: Optional[str] = None,
                     history_filename: Optional[str] = None,
                     plots_dir: Optional[str] = None,
                     save_best_model: bool = False,
                     save_each_epoch: bool = False) -> Dict[str, List[float]]:
        """
            Performs the train for a given number of epochs.
        """
        print(f"TRAINING... \n\ton device: {device} \n\twith loss: {self.criterion}\n")

        if config_train.USE_WANDB:
            wandb.login(key=creds.api_key_wandb)
            wandb.init(
                name=model_dir,  # keeping only the last part of the model_dir (it stores all the viable information)
                project=f"fl-mri")

        if not isinstance(self.criterion, Callable):
            raise TypeError(f"Loss function (criterion) has to be callable. It is {type(self.criterion)} which is not.")

        if any([history_filename, plots_dir, filename]):
            fop.try_create_dir(model_dir)
            print(f"\tModel, history and plots will be saved to {model_dir}")
        else:
            print(f"\tWARNING: Neither model, history nor plots from the training process will be saved!")

        val_metric_names = [f"val_{m_name}" for m_name in config_train.METRICS]

        history = {m_name: [] for m_name in config_train.METRICS}
        history.update({m_name: [] for m_name in val_metric_names})

        if save_best_model:
            best_loss = 1000.0

        if plots_dir is not None:
            plots_path = path.join(model_dir, plots_dir)
            fop.try_create_dir(plots_path)
        else:
            plots_path = None

        for epoch in range(epochs):
            print(f"\tEPOCH: {epoch + 1}/{epochs}")

            epoch_metrics = self._train_one_epoch(trainloader, optimizer)

            for metric in config_train.METRICS:
                history[metric].append(epoch_metrics[metric])

            print("\tVALIDATION...")
            if validationloader is not None:
                val_metric = self.evaluate(validationloader, plots_path, f"ep{epoch}.jpg")

                for metric in val_metric_names:
                    # trimming after val_ to get only the metric name since it is provided by the
                    history[metric].append(val_metric[metric])

                if save_best_model:
                    if val_metric["val_loss"] < best_loss:
                        print(
                            f"\tModel form epoch {epoch} taken as the best one.\n\tIts loss {val_metric['val_loss']:.3f} is better than current best loss {best_loss:.3f}.")
                        best_loss = val_metric["val_loss"]
                        best_model = self.state_dict()

                if config_train.USE_WANDB:
                    wandb.log(val_metric)

            if config_train.USE_WANDB:
                wandb.log(epoch_metrics)

            if save_each_epoch:
                self.save(model_dir, f"model-ep{epoch}.pth")

            if history_filename is not None:
                with open(path.join(model_dir, history_filename), 'wb') as file:
                    pickle.dump(history, file)

        # saving best model
        if save_best_model:
            torch.save(best_model, path.join(model_dir, "best_model.pth"))

        print("\tAll epochs finished.\n")

        if filename is not None:
            self.save(model_dir, filename)

        return history

    def evaluate(self,
                testloader: DataLoader,
                plots_path: Optional[str] = None,
                plot_filename: Optional[str] = None,
                compute_std: bool = False,
                wanted_metrics: Optional[List[str]] = None,
                min_mask_pixel_in_batch: int = 9,
                save_preds_dir: Optional[str] = None,
                plot_metrics_distribution: bool = False,
                low_ssim_value: float = float('inf')) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
        print(f"\tON DEVICE: {device} \n\tWITH LOSS: {self.criterion}\n")

        self.eval()

        if not isinstance(self.criterion, Callable):
            raise TypeError(f"Loss function (criterion) has to be callable. It is {type(self.criterion)} which is not.")
        if compute_std and testloader.batch_size != 1:
            raise ValueError("The computations will result in wrong results! Batch size should be 1 if `compute_std=True`.")

        n_steps = 0
        n_skipped = 0
        metrics = {metric_name: self.available_metrics[metric_name] for metric_name in config_train.METRICS}

        if wanted_metrics:
            metrics = {metric_name: metric_obj for metric_name, metric_obj in metrics.items() if
                       metric_name in wanted_metrics}

        metrics = {f"val_{name}": metric for name, metric in metrics.items()}

        if save_preds_dir:
            fop.try_create_dir(save_preds_dir)
        if plots_path:
            fop.try_create_dir(plots_path)


        metrics_values = {m_name: [] for m_name in metrics.keys()}
        with torch.no_grad():
            for batch_index, batch in enumerate(testloader):
                # loading the input and target images
                images_cpu, targets_cpu = batch[0], batch[1]

                # the is possibility that a metric will require a mask e.g. brain mask
                # then it is should be loaded from the dataloader
                if testloader.dataset.metric_mask:
                    metric_mask_cpu = batch[2]
                    metric_mask = metric_mask_cpu.to(config_train.DEVICE)

                images = images_cpu.to(config_train.DEVICE)
                targets = targets_cpu.to(config_train.DEVICE)

                # utilizing the network
                predictions = self(images)

                # saving all the predictions
                if save_preds_dir:
                    current_batch_size = images.shape[0]

                    if batch_index == 0:  # getting the batch size in the first round
                        batch_size = current_batch_size

                    for img_index in range(current_batch_size):  # iterating over current batch size (number of images)
                        # retrieving the name of the current slice
                        patient_slice_name = \
                            testloader.dataset.images[img_index + batch_index * batch_size].split(path.sep)[-1]
                        pred_filepath = path.join(save_preds_dir, patient_slice_name)

                        # saving the current image to the declared directory with the same name as the input image name
                        np.save(pred_filepath, predictions[img_index].cpu().numpy())

                # calculating metrics
                for metric_name, metric_obj in metrics.items():
                    if isinstance(metric_obj, custom_metrics.LossWithProximalTerm):
                        metric_obj = metric_obj.base_loss_fn

                    if isinstance(metric_obj, custom_metrics.ZoomedSSIM):
                        if testloader.dataset.metric_mask:
                            mask_size = torch.sum(metric_mask)
                            if mask_size > min_mask_pixel_in_batch:
                                metric_value = metric_obj(predictions, targets, metric_mask)
                            else:
                                logging.log(logging.WARNING,
                                            f"The batch number {batch_index} is skipped for `ZoomedSSIM` calculations"
                                            f"due to small mask ({mask_size} pixels). It can affect the metric result. ")
                                n_skipped += 1
                                break
                        else:
                            ValueError(
                                f"Dataloader should be able to load the mask of the image to compute: {metric_name}. "
                                f"Provide `metric_mask_dir` in the dataset `MRIDatasetNumpySlices` object initialization to use this metric.")

                    else:
                        metric_value = metric_obj(predictions, targets)

                        if plots_path:
                            if isinstance(metric_obj, custom_metrics.MaskedSSIM) and metric_value.item() < low_ssim_value:
                                slice_filename = testloader.dataset.images[img_index + batch_index * batch_size].split(path.sep)[-1]
                                print(f"For slice `{slice_filename}` maseked SSIM is low ({metric_value:.2f}), saving this slice.")
                                filepath = path.join(plots_path, f"slice{batch_index}_ssim{metric_value:.2f}.jpg")

                                visualization.plot_single_data_sample(
                                    [images.to('cpu'), targets.to('cpu'), predictions.to('cpu').detach()],
                                    filepath=filepath)

                    metrics_values[metric_name].append(metric_value.item())

                n_steps += 1

        if plots_path and plot_filename:
            filepath = path.join(plots_path, plot_filename)
            visualization.plot_batch([images.to('cpu'), targets.to('cpu'), predictions.to('cpu').detach()],
                                     filepath=filepath)

        if plot_metrics_distribution:
            histograms_dir_path = path.join(plots_path, "histograms")
            fop.try_create_dir(histograms_dir_path)

            # Plot and save histograms
            for key, values in metrics_values.items():
                plt.figure()  # Create a new figure
                plt.hist(values, bins=100, color='blue', alpha=0.7)
                plt.title(f"Histogram of {key}")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

                # Save to file
                output_path = path.join(histograms_dir_path, f"{key}_histogram.png")
                plt.savefig(output_path)
                plt.close()  # Close the figure to free up memory
                print(f"Saved histogram for {key} to {output_path}")

            print("All histograms saved.")

        averaged_metrics, std_metrics = self._compute_average_std_metric(metrics_values, n_steps, n_skipped)
        metrics_str = custom_metrics.metrics_to_str(averaged_metrics, starting_symbol="\n\t")

        print(f"\tFor evaluation set: {metrics_str}\n")

        if compute_std:
            return averaged_metrics, std_metrics
        else:
            return averaged_metrics

    @staticmethod
    def _compute_average_std_metric(metrics_values, n_steps, n_skipped):
        """
        Computes the average for each of the metrics using the sum and the number of steps. 
        Treats specially the ZoomedSSIM metrics since there are potential skips. 
        """
        averaged_metrics = {}
        std_metrics = {}
        for metric_name, metric_values in metrics_values.items():
            numpy_metrics_values = np.array(metric_values)
            if "zoomed_ssim" in metric_name:
                if n_skipped == n_steps:
                    logging.log(logging.WARNING, f"All the mask in the provided dataset are zeros."
                                                 "\nNone ZoomedSSIM values were computed. Result assigned to None")

                    averaged_metrics[metric_name] = None
                denominator = n_steps - n_skipped
            else:
                denominator = n_steps

            averaged_metrics[metric_name] = numpy_metrics_values.sum() / denominator
            std_metrics[metric_name] = numpy_metrics_values.std()

        return averaged_metrics, std_metrics


class DoubleConv(nn.Module):
    """Double convolution block with optional normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 normalization: enums.NormalizationType, 
                 mid_channels: Optional[int] = None) -> None:
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        # choosing between one of three possible normalization types
        if normalization == config_train.NormalizationType.BN:
            self.norm1 = nn.BatchNorm2d(mid_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif normalization == config_train.NormalizationType.GN:
            self.norm1 = nn.GroupNorm(config_train.N_GROUP_NORM, mid_channels)
            self.norm2 = nn.GroupNorm(config_train.N_GROUP_NORM, out_channels)
        else:
            self.norm1 = None
            self.norm2 = None

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm1:
            x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.norm2:
            x = self.norm2(x)
        x = self.relu(x)

        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, 
                 normalization: enums.NormalizationType) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, normalization)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, 
                 normalization: enums.NormalizationType, 
                 bilinear: bool = True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, normalization, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, normalization)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final convolution layer with sigmoid activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))
