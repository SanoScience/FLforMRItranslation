from os import path
from itertools import chain
import pickle
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import src.loss_functions
from configs import config_train
from src import visualization, loss_functions, files_operations as fop

device = config_train.DEVICE
batch_print_freq = config_train.BATCH_PRINT_FREQ
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
mse = nn.MSELoss()


class UNet(nn.Module):
    # TODO: test with bilinear = False

    def __init__(self, bilinear=False, normalization=config_train.NORMALIZATION):
        super(UNet, self).__init__()

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

    def _train_one_epoch(self, trainloader, optimizer, criterion, batch_print_frequency, prox_loss):
        # TODO: enable choosing metrics
        def reset_dict(dictionary: dict):
            return {key: 0.0 for key in dictionary.keys()}

        metrics = {"loss": criterion, "ssim": ssim, "pnsr": psnr, "mse": mse}

        epoch_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}
        total_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}

        n_batches = len(trainloader)

        start = time.time()
        n_train_steps = 0

        if prox_loss:
            global_params = [val.detach().clone() for val in self.parameters()]

        for index, data in enumerate(trainloader):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = self(images)

            if prox_loss:
                loss = criterion(predictions, targets, self.parameters(), global_params)
            else:
                loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            for metric_name, metrics_obj in metrics.items():
                metric_value = metrics_obj(predictions, targets)
                total_metrics[metric_name] += metric_value.item()
                epoch_metrics[metric_name] += metric_value.item()

            n_train_steps += 1

            if index % batch_print_frequency == batch_print_frequency - 1:

                divided_batch_metrics = {metric_name: total_value/batch_print_frequency for metric_name, total_value in total_metrics.items()}
                metrics_str = loss_functions.metrics_to_str(divided_batch_metrics, starting_symbol="\t")

                print(f'\tbatch {(index + 1)} out of {n_batches}{metrics_str}')

                total_metrics = reset_dict(total_metrics)

        divided_epoch_metrics = {metric_name: metric_value / n_train_steps for metric_name, metric_value in epoch_metrics.items()}
        metrics_epoch_str = loss_functions.metrics_to_str(divided_epoch_metrics, starting_symbol="")

        print(f"\n\tTime exceeded: {time.time() - start:.1f}\n\tEpoch metrics:{metrics_epoch_str}")
        print()

        return divided_epoch_metrics

    def perform_train(self,
                      trainloader: DataLoader,
                      optimizer,
                      criterion,
                      epochs,
                      prox_loss=False,
                      validationloader=None,
                      model_dir=config_train.TRAINED_MODEL_SERVER_DIR,
                      filename=None,
                      history_filename=None,
                      plots_dir=None):
        print(f"TRAINING... \n\ton device: {device} \n\twith loss: {criterion}\n")

        if not isinstance(criterion, Callable):
            raise TypeError(f"Loss function (criterion) has to be callable. It is {type(criterion)} which is not.")

        if any([history_filename, plots_dir, filename]):
            fop.try_create_dir(model_dir)
            print(f"\tModel, history and plots will be saved to {model_dir}")
        else:
            print(f"\tWARNING: Neither model, history nor plots from the training process will be saved!")

        n_batches = len(trainloader)

        if batch_print_freq > n_batches:
            print_freq = n_batches - 2  # tbh not sure if this -2 is needed
        else:
            print_freq = batch_print_freq

        val_metric_names = [f"val_{m_name}" for m_name in config_train.METRICS]

        history = {m_name: [] for m_name in config_train.METRICS}
        history.update({m_name: [] for m_name in val_metric_names})

        if plots_dir is not None:
            plots_path = path.join(model_dir, plots_dir)
            fop.try_create_dir(plots_path)

        self.train()

        for epoch in range(epochs):
            print(f"\tEPOCH: {epoch + 1}/{epochs}")

            epoch_metrics = self._train_one_epoch(trainloader, optimizer, criterion, print_freq, prox_loss)

            for metric in config_train.METRICS:
                history[metric].append(epoch_metrics[metric])

            print("\tVALIDATION...")
            if validationloader is not None:
                val_metric = self.evaluate(validationloader, criterion, model_dir, plots_dir, f"ep{epoch}")

                for metric in val_metric_names:
                    # trimming after val_ to get only the metric name since it is provided by the
                    history[metric].append(val_metric[metric[len("val_"):]])

        print("\tAll epochs finished.\n")

        # saving
        if history_filename is not None:
            with open(path.join(model_dir, history_filename), 'wb') as file:
                pickle.dump(history, file)

        if filename is not None:
            self.save(model_dir, filename)

        return history

    def evaluate(self, testloader, criterion, model_dir=config_train.TRAINED_MODEL_SERVER_DIR, plots_dir=None, plot_filename=None):
        if isinstance(criterion, src.loss_functions.LossWithProximalTerm):
            criterion = criterion.base_loss_fn

        print(f"\tON DEVICE: {device} \n\tWITH LOSS: {criterion}\n")

        if not isinstance(criterion, Callable):
            raise TypeError(f"Loss function (criterion) has to be callable. It is {type(criterion)} which is not.")

        n_test_steps = 0
        metrics = {"loss": criterion, "ssim": ssim, "pnsr": psnr, "mse": mse}
        metrics_values = {m_name: 0.0 for m_name in config_train.METRICS}

        self.eval()
        with torch.no_grad():
            for images_cpu, targets_cpu in testloader:
                images = images_cpu.to(device)
                targets = targets_cpu.to(device)

                predictions = self.__call__(images)

                for metric_name, metrics_obj in metrics.items():
                    metric_value = metrics_obj(predictions, targets)
                    metrics_values[metric_name] += metric_value.item()

                n_test_steps += 1

        divided_metrics = {metric_name: metric_value / n_test_steps for metric_name, metric_value in metrics_values.items()}
        metrics_str = loss_functions.metrics_to_str(divided_metrics, starting_symbol="\n\t")

        print(f"\tFor evaluation set: {metrics_str}\n")

        if plots_dir is not None:
            directory = path.join(model_dir, plots_dir)
            fop.try_create_dir(directory)
            filepath = path.join(directory, f"{plot_filename}.jpg")
            # maybe cast to cpu ?? still dunno if needed
            visualization.plot_batch([images_cpu, targets_cpu, predictions.to('cpu').detach()], filepath=filepath)

        return divided_metrics

    def save(self, dir_name: str, filename: str = None, create_dir=True):
        if filename is None:
            filename = "model"

        if not isinstance(dir_name, str):
            raise TypeError(f"Given directory name {dir_name} has wrong type: {type(dir_name)}.")

        if create_dir:
            fop.try_create_dir(dir_name)

        if not filename.endswith(".pth"):
            filename += ".pth"

        filepath = f"{dir_name}/{filename}"
        torch.save(self.state_dict(), filepath)

        print("Model saved to: ", filepath)

    def __str__(self):
        return f"UNet(batch_norm={config_train.NORMALIZATION})"


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
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OldUNet(nn.Module):
    def __init__(self, bilinear=False):
        super(OldUNet, self).__init__()
        self.bilinear = bilinear

        self.inc = (OldDoubleConv(1, 64))
        self.down1 = (OldDown(64, 128))
        self.down2 = (OldDown(128, 256))
        self.down3 = (OldDown(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (OldDown(512, 1024 // factor))
        self.up1 = (OldUp(1024, 512 // factor, bilinear))
        self.up2 = (OldUp(512, 256 // factor, bilinear))
        self.up3 = (OldUp(256, 128 // factor, bilinear))
        self.up4 = (OldUp(128, 64, bilinear))
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


class OldDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OldDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            OldDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OldUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = OldDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = OldDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


def train(model,
          trainloader,
          validationloader,
          optimizer,
          criterion,
          epochs,
          filename=None,
          history_filename="history.pkl",
          plots_dir=None):
    # TODO: transform from config to local vars

    print(f"Training \non device: {config_train.DEVICE} \nwith loss: {criterion})...\n")

    model_dir = config_train.TRAINED_MODEL_SERVER_DIR
    fop.try_create_dir(model_dir)
    print(f"Created directory {model_dir}")

    n_batches = len(trainloader)

    if n_batches < config_train.BATCH_PRINT_FREQ:
        loss_print_freq = n_batches - 2  # tbh not sure if this -2 is needed
    else:
        loss_print_freq = config_train.BATCH_PRINT_FREQ

    train_losses = []
    train_ssims = []
    val_losses = []
    val_ssims = []

    n_train_steps = len(trainloader.dataset) // config_train.BATCH_SIZE
    n_val_steps = len(validationloader.dataset) // config_train.BATCH_SIZE

    if plots_dir is not None:
        plots_path = path.join(model_dir, plots_dir)
        fop.try_create_dir(plots_path)

    for epoch in range(epochs):
        print("EPOCH: ", epoch + 1)

        running_loss, total_ssim = 0.0, 0.0
        epoch_loss, epoch_ssim = 0.0, 0.0

        start = time.time()

        for index, data in enumerate(trainloader):
            images, targets = data
            images = images.to(config_train.DEVICE)
            targets = targets.to(config_train.DEVICE)

            optimizer.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()

            optimizer.step()

            # predictions_double = predictions.double()
            # targets_double = targets.double()

            # print(f"Predictions shape: {predictions_double.shape} type: {predictions_double.type()}")
            # print(f"Targets shape: {targets_double.shape} type: {targets_double.type()}")

            ssim_value = ssim(predictions, targets)

            running_loss += loss.item()
            total_ssim += ssim_value.item()

            epoch_loss += loss.item()
            epoch_ssim += ssim_value.item()

            if index % loss_print_freq == loss_print_freq - 1:

                print(f'batch {(index + 1)} out of {n_batches}\t'
                      f'loss: {running_loss / loss_print_freq:.3f} '
                      f'ssim {total_ssim / loss_print_freq:.3f}')

                running_loss = 0.0
                total_ssim = 0.0

        epoch_loss /= n_train_steps
        epoch_ssim /= n_train_steps
        print(f"\nTime exceeded: {time.time() - start:.1f} "
              f"epoch loss: {epoch_loss:.3f} ssim: {epoch_ssim:.3f}")
        print()

        train_ssims.append(epoch_ssim)
        train_losses.append(epoch_loss)

        print("Validation set in progress...")
        val_loss = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for images_cpu, targets_cpu in validationloader:
                images = images_cpu.to(config_train.DEVICE)
                targets = targets_cpu.to(config_train.DEVICE)

                predictions = model(images)
                loss = criterion(predictions, targets)

                val_loss += loss.item()
                val_ssim += ssim(predictions, targets).item()

        val_loss /= n_val_steps
        val_ssim /= n_val_steps
        print(f"For validation set: val_loss: {val_loss:.3f} "
              f"val_ssim: {val_ssim:.3f}")

        val_ssims.append(val_ssim)
        val_losses.append(val_loss)

        if plots_dir is not None:
            filepath = path.join(model_dir, plots_dir, f"ep{epoch}.jpg")
            # maybe cast to cpu ?? still dunno if needed
            visualization.plot_batch([images, targets, predictions.to('cpu').detach()], filepath=filepath)

    print("\nEnd of this round.")

    history = {"loss": train_losses, "ssim": train_ssims, "val_loss": val_losses, "val_ssim": val_ssims}

    # saving
    if history_filename is not None:
        with open(path.join(model_dir, history_filename), 'wb') as file:
            pickle.dump(history, file)

    if filename is not None:
        model.save(model_dir, filename)

    return history


def test(model, testloader, criterion):
    print(f"Testing \non device: {config_train.DEVICE} \nwith loss: {criterion})...\n")
    n_steps = 0

    total_loss = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for images_cpu, targets_cpu in testloader:
            images = images_cpu.to(config_train.DEVICE)
            targets = targets_cpu.to(config_train.DEVICE)

            predictions = model(images)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            total_ssim += ssim(predictions, targets).item()

            n_steps += 1

    return total_loss / n_steps, total_ssim / n_steps