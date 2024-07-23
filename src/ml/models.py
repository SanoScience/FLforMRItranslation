from os import path
import pickle
import wandb
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.classification import Dice, BinaryJaccardIndex

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
dice_score = Dice().to(device)
jaccard_index = BinaryJaccardIndex().to(device)
# zoomed_ssim = loss_functions.ZoomedSSIM()
# qilv = loss_functions.QILV(use_mask=False)

class UNet(nn.Module):
    """
        UNet model class used for federated learning.
        Consists the methods to train and evaluate.
        Allows two different normalizations: 
            - standard BatchNormalization
            - GroupNorm (the number of groups specified in the config)
    """
    def __init__(self, criterion=None, bilinear=False, normalization=config_train.NORMALIZATION):
        super(UNet, self).__init__()

        self.criterion = criterion
        self.bilinear = bilinear

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

    def _train_one_epoch(self, trainloader, optimizer):
        """
        Method used by perform_train(). Does one iteration of training.
        """
        metrics = {"loss": self.criterion, "ssim": ssim, "pnsr": psnr, "mse": mse, "masked_mse": masked_mse, "relative_error": relative_error, "dice": dice_score, "jaccard": jaccard_index}

        epoch_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}
        total_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}

        n_batches = len(trainloader)

        start = time.time()
        n_train_steps = 0

        # running_loss, total_ssim = 0.0, 0.0
        # epoch_loss, epoch_ssim = 0.0, 0.0

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

            # loss.requires_grad = True  # This is ugly af, but not a better solution found yet
            loss.backward()
            optimizer.step()

            for metric_name, metric_object in metrics.items():
                if metric_name == "loss":
                    metric_value = loss
                elif metric_name == "dice":
                    metric_value = metric_object(predictions, targets.int())
                else:
                    metric_value = metric_object(predictions, targets)
                total_metrics[metric_name] += metric_value.item()
                epoch_metrics[metric_name] += metric_value.item()

            n_train_steps += 1

            if index % batch_print_frequency == batch_print_frequency - 1:
                divided_batch_metrics = {metric_name: total_value/batch_print_frequency for metric_name, total_value in total_metrics.items()}
                metrics_str = custom_metrics.metrics_to_str(divided_batch_metrics, starting_symbol="\t")
                print(f'\tbatch {(index + 1)} out of {n_batches}\t\t{metrics_str}')

                total_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}

        averaged_epoch_metrics = {metric_name: metric_value / n_train_steps for metric_name, metric_value in epoch_metrics.items()}
        metrics_epoch_str = custom_metrics.metrics_to_str(averaged_epoch_metrics, starting_symbol="")

        print(f"\n\tTime exceeded: {time.time() - start:.1f}")
        print(f"\tEpoch metrics: {metrics_epoch_str}")

        return averaged_epoch_metrics

    def perform_train(self,
                      trainloader,
                      optimizer,
                      epochs,
                      validationloader=None,
                      model_dir=config_train.TRAINED_MODEL_SERVER_DIR,
                      filename=None,
                      history_filename=None,
                      plots_dir=None,
                      save_best_model=False, 
                      save_each_epoch=False):
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
                        print(f"\tModel form epoch {epoch} taken as the best one.\n\tIts loss {val_metric['val_loss']} is better than current best loss {best_loss}.")
                        best_loss = val_metric["val_loss"]
                        best_model = self.state_dict()

                if config_train.USE_WANDB:
                    wandb.log(val_metric)

            if config_train.USE_WANDB:
                wandb.log(epoch_metrics)

            if save_each_epoch:
                self.save(model_dir, f"model-ep{epoch}.pth")       

            # saving
            if save_best_model:
                torch.save(best_model, path.join(model_dir, "best_model.pth"))

            if history_filename is not None:
                with open(path.join(model_dir, history_filename), 'wb') as file:
                    pickle.dump(history, file)

        print("\tAll epochs finished.\n")

        if filename is not None:
            self.save(model_dir, filename)

        return history

    def evaluate(self, testloader, plots_path=None, plot_filename=None, evaluate=True, wanted_metrics=None, save_preds_dir=None):
        print(f"\tON DEVICE: {device} \n\tWITH LOSS: {self.criterion}\n")

        if not isinstance(self.criterion, Callable):
            raise TypeError(f"Loss function (criterion) has to be callable.It is {type(self.criterion)} which is not.")

        n_steps = 0

        metrics = {"loss": self.criterion, "ssim": ssim, "pnsr": psnr, "mse": mse, "masked_mse": masked_mse, "masked_ssim": masked_ssim, "relative_error": relative_error, "dice": dice_score, "jaccard": jaccard_index}

        if wanted_metrics:
            metrics = {metric_name: metric_obj for metric_name, metric_obj in metrics.items() if metric_name in wanted_metrics}

        if evaluate:
            metrics = {f"val_{name}": metric for name, metric in metrics.items()}

        if save_preds_dir:
            fop.try_create_dir(save_preds_dir)

        metrics_values = {m_name: 0.0 for m_name in metrics.keys()}
        with torch.no_grad():
            for batch_index, (images_cpu, targets_cpu, metric_mask_cpu) in enumerate(testloader):
                # loading the input and target images
                images = images_cpu.to(config_train.DEVICE)
                targets = targets_cpu.to(config_train.DEVICE)
                metric_mask = metric_mask_cpu.to(config_train.DEVICE)

                # utilizing the network
                predictions = self(images)

                # saving all the predictions
                if save_preds_dir:
                    current_batch_size = images.shape[0]

                    if batch_index == 0:  # getting the batch size in the first round
                        batch_size = current_batch_size

                    for img_index in range(current_batch_size):  # iterating over current batch size (number of images)
                        # retrieving the name of the current slice
                        patient_slice_name = testloader.dataset.images[img_index+batch_index*batch_size].split(path.sep)[-1]
                        pred_filepath = path.join(save_preds_dir, patient_slice_name)

                        # saving the current image to the declared directory with the same name as the input image name
                        # print(pred_filepath)
                        np.save(pred_filepath, predictions[img_index].cpu().numpy())

                # calculating metrics
                for metric_name, metrics_obj in metrics.items():
                    if isinstance(metrics_obj, custom_metrics.LossWithProximalTerm):
                        metrics_obj = metrics_obj.base_loss_fn
                    if isinstance(metrics_obj, custom_metrics.ZoomedSSIM):
                        metric_value = metrics_obj(predictions, targets, metric_mask)
                    else:
                        metric_value = metrics_obj(predictions, targets)

                    print(f"{metric_name}: ", metric_value)
                    metrics_values[metric_name] += metric_value.item()

                n_steps += 1

        if plots_path:
            fop.try_create_dir(plots_path)  # pretty sure this is not needed (only makes warnings)
            filepath = path.join(plots_path, plot_filename)
            # maybe cast to cpu ?? still dunno if needed
            visualization.plot_batch([images.to('cpu'), targets.to('cpu'), predictions.to('cpu').detach()],
                                     filepath=filepath)

        averaged_metrics = {metric_name: metric_value / n_steps for metric_name, metric_value in metrics_values.items()}
        metrics_str = custom_metrics.metrics_to_str(averaged_metrics, starting_symbol="\n\t")

        print(f"\tFor evaluation set: {metrics_str}\n")

        return averaged_metrics


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, mid_channels=None):
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

    def __init__(self, in_channels, out_channels, normalization):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, normalization)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, normalization, bilinear=True):
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
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))
