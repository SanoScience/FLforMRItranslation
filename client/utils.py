import os.path
import pickle
import time

import torch

from common import datasets, config_train, utils
from torchmetrics.image import StructuralSimilarityIndexMeasure

# TODO: metrics as a train parameter instead of this
ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))


def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    trainset = datasets.MRIDatasetNumpySlices([train_dir])
    testset = datasets.MRIDatasetNumpySlices([test_dir])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config_train.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config_train.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


def train(model,
          trainloader,
          validationloader,
          optimizer,
          epochs,
          filename=None,
          history_filename="history.pkl",
          plots_dir=None):
    # TODO: transform from config to local vars

    print(f"Training (on device: {config_train.DEVICE})...\n")

    model_dir = config_train.TRAINED_MODEL_CLIENT_DIR
    utils.try_create_dir(model_dir)
    print(f"Created directory {model_dir}")

    n_batches = len(trainloader)

    if n_batches < config_train.LOSS_PRINT_FREQ:
        loss_print_freq = n_batches - 2  # tbh not sure if this -2 is needed
    else:
        loss_print_freq = config_train.LOSS_PRINT_FREQ

    train_losses = []
    train_ssims = []
    val_losses = []
    val_ssims = []

    if plots_dir is not None:
        plots_path = os.path.join(model_dir, plots_dir)
        utils.try_create_dir(plots_path)

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
            loss = config_train.CRITERION(predictions, targets)
            loss.backward()

            optimizer.step()

            ssim_value = ssim(predictions, targets).item()

            running_loss += loss.item()
            total_ssim += ssim_value

            epoch_loss += loss.item()
            epoch_ssim += ssim_value

            if index % loss_print_freq == loss_print_freq - 1:

                print(f'batch {(index + 1)} out of {n_batches}\t'
                      f'loss: {running_loss / loss_print_freq:.3f} '
                      f'ssim {total_ssim / loss_print_freq:.3f}')

                running_loss = 0.0
                total_ssim = 0.0

        print(f"\nTime exceeded: {time.time() - start:.1f} epoch loss: {epoch_loss:.3f} ssim: {epoch_ssim:.3f}")
        print()

        if plots_dir is not None:
            filepath = os.path.join(model_dir, plots_dir, f"ep{epoch}.jpg")
            # maybe cast to cpu ??
            utils.plot_predicted_batch(images, targets, predictions, filepath=filepath)

        train_ssims.append(epoch_ssim)
        train_losses.append(epoch_loss)

        print("Validation set in progress...")
        val_loss = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for images, targets in validationloader:
                images = images.to(config_train.DEVICE)
                targets = targets.to(config_train.DEVICE)

                predictions = model(images)
                loss = config_train.CRITERION(predictions, targets)

                val_loss += loss.item()
                val_ssim += ssim(predictions, targets).item()

        print(f"For validation set: val_loss: {val_loss:.3f} val_ssim: {val_ssim:.3f}")

        val_ssims.append(val_ssim)
        val_losses.append(val_loss)

    print("\nEnd of this round.")

    history = {"loss": train_ssims, "ssim": train_ssims, "val_loss": val_losses, "val_ssim": val_ssims}

    # saving
    if history_filename is not None:
        with open(os.path.join(model_dir, history_filename), 'wb') as file:
            pickle.dump(history, file)

    if filename is not None:
        model.save(model_dir, filename)

    return history


def test(model, testloader):
    print("Testing...\n")

    total_ssim = 0.0
    loss = 0.0

    with torch.no_grad():
        for images, targets in testloader:
            predicted = model(images)

            loss += config_train.CRITERION(predicted, targets).item()
            total_ssim += ssim(predicted, targets)

    return loss, total_ssim / len(testloader.dataset)
