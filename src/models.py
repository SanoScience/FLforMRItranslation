from os import path
import pickle
import time
from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

import src.loss_functions
from configs import config_train
from src import visualization, files_operations as fop


class UNet(nn.Module):
    def __init__(self, bilinear=False, batch_normalization=False):
        super(UNet, self).__init__()
        self.batch_normalization = batch_normalization
        self.bilinear = bilinear

        self.inc = (DoubleConv(1, 64, batch_normalization))
        self.down1 = (Down(64, 128, batch_normalization))
        self.down2 = (Down(128, 256, batch_normalization))
        self.down3 = (Down(256, 512, batch_normalization))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, batch_normalization))
        self.up1 = (Up(1024, 512 // factor, batch_normalization, bilinear))
        self.up2 = (Up(512, 256 // factor, batch_normalization, bilinear))
        self.up3 = (Up(256, 128 // factor, batch_normalization, bilinear))
        self.up4 = (Up(128, 64, batch_normalization, bilinear))
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

    def save(self, dir_name: str, filename=None, create_dir=True):
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

    def __repr__(self):
        return f"UNet(batch_norm={self.batch_normalization})"

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, batch_normalization: bool, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        if batch_normalization:
            layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                      nn.BatchNorm2d(mid_channels),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
        else:
            layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                      nn.ReLU(inplace=True)]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, batch_normalization):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, batch_normalization)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, batch_normalization, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, batch_normalization)

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


model_dir = config_train.TRAINED_MODEL_SERVER_DIR
device = config_train.DEVICE
batch_print_freq = config_train.BATCH_PRINT_FREQ
ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)


def _train_one_epoch(model, trainloader, optimizer, criterion, batch_print_frequency, prox_loss):
    running_loss, total_ssim = 0.0, 0.0
    epoch_loss, epoch_ssim = 0.0, 0.0

    n_batches = len(trainloader)

    start = time.time()
    n_train_steps = 0

    if prox_loss:
        global_params = [val.detach().clone() for val in model.parameters()]

    for index, data in enumerate(trainloader):
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        if prox_loss:
            loss = criterion(predictions, targets, model.parameters(), global_params)
        else:
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

        n_train_steps += 1

        if index % batch_print_frequency == batch_print_frequency - 1:
            print(f'batch {(index + 1)} out of {n_batches}\t'
                  f'loss: {running_loss / batch_print_frequency:.3f} '
                  f'ssim {total_ssim / batch_print_frequency:.3f}')

            running_loss = 0.0
            total_ssim = 0.0

    epoch_loss /= n_train_steps
    epoch_ssim /= n_train_steps
    print(f"\nTime exceeded: {time.time() - start:.1f} "
          f"epoch loss: {epoch_loss:.3f} ssim: {epoch_ssim:.3f}")
    print()

    return epoch_loss, epoch_ssim


def train(model,
          trainloader,
          optimizer,
          criterion,
          epochs,
          prox_loss=False,
          validationloader=None,
          filename=None,
          history_filename=None,
          plots_dir=None):
    print(f"Training... \n\ton device: {device} \n\twith loss: {criterion}\n")

    if not isinstance(criterion, Callable):
        raise TypeError(f"Loss function (criterion) has to be callable. It is {type(criterion)} which is not.")

    if any([history_filename, plots_dir, filename]):
        fop.try_create_dir(model_dir)
        print(f"Model, history and plots will be saved to {model_dir}")
    else:
        print(f"Neither model, history nor plots from the training process will be saved!")


    n_batches = len(trainloader)

    if batch_print_freq > n_batches:
        batch_print_frequency = n_batches - 2  # tbh not sure if this -2 is needed
    else:
        batch_print_frequency = batch_print_freq

    train_losses = []
    train_ssims = []
    val_losses = []
    val_ssims = []

    if plots_dir is not None:
        plots_path = path.join(model_dir, plots_dir)
        fop.try_create_dir(plots_path)

    model.train()

    for epoch in range(epochs):
        print(f"EPOCH: {epoch + 1}/{epochs}")

        epoch_loss, epoch_ssim = _train_one_epoch(model, trainloader, optimizer, criterion, batch_print_frequency,
                                                  prox_loss)

        train_ssims.append(epoch_ssim)
        train_losses.append(epoch_loss)

        if validationloader is not None:
            val_loss, val_ssim = evaluate(model, validationloader, criterion, plots_dir, epoch)

            val_ssims.append(val_ssim)
            val_losses.append(val_loss)

    print("\nEnd of this training round.\n\n")

    history = {"loss": train_losses, "ssim": train_ssims, "val_loss": val_losses, "val_ssim": val_ssims}

    # saving
    if history_filename is not None:
        with open(path.join(model_dir, history_filename), 'wb') as file:
            pickle.dump(history, file)

    if filename is not None:
        model.save(model_dir, filename)

    return history


def evaluate(model, testloader, criterion, plots_dir=None, epoch=0):
    if isinstance(criterion, src.loss_functions.LossWithProximalTerm):
        criterion = criterion.base_loss_fn

    print(f"Testing... \n\tON DEVICE: {device} \n\tWITH LOSS: {criterion}\n")

    if not isinstance(criterion, Callable):
        raise TypeError(f"Loss function (criterion) has to be callable. It is {type(criterion)} which is not.")

    n_test_steps = 0

    test_loss = 0.0
    test_ssim = 0.0

    model.eval()
    with torch.no_grad():
        for images_cpu, targets_cpu in testloader:
            images = images_cpu.to(device)
            targets = targets_cpu.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            test_loss += loss.item()
            test_ssim += ssim(predictions, targets).item()

            n_test_steps += 1

    test_loss /= n_test_steps
    test_ssim /= n_test_steps
    print(f"For evaluation set: test_loss: {test_loss:.3f} "
          f"test_ssim: {test_ssim:.3f}")

    if plots_dir is not None:
        filepath = path.join(model_dir, plots_dir, f"ep{epoch}.jpg")
        # maybe cast to cpu ?? still dunno if needed
        visualization.plot_batch([images_cpu, targets_cpu, predictions.to('cpu').detach()], filepath=filepath)

    return test_loss, test_ssim

# def test(model, testloader: DataLoader, criterion, plots_dir=None, epoch=None) -> Tuple[float, float]:
#     print(f"Testing \non device: {device} \nwith loss: {criterion})...\n")
#
#     if not isinstance(criterion, Callable):
#         raise TypeError(f"Loss function (criterion) has to be callable. It is {type(criterion)} which is not.")
#
#     n_steps = 0
#
#     total_loss = 0.0
#     total_ssim = 0.0
#     with torch.no_grad():
#         for images_cpu, targets_cpu in testloader:
#             images = images_cpu.to(device)
#             targets = targets_cpu.to(device)
#
#             predictions = model(images)
#             loss = criterion(predictions, targets)
#
#             total_loss += loss.item()
#             total_ssim += ssim(predictions, targets).item()
#
#             n_steps += 1
#
#     loss_value, ssim_value = total_loss / n_steps, total_ssim / n_steps
#
#     print(f"Test loss: {loss_value:.4f} ssim: {ssim_value:.4f}")
#
#     return loss_value, ssim_value
