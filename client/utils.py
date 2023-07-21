import os.path
import time

import torch

from common import datasets, config_train, utils
from torchmetrics.image import StructuralSimilarityIndexMeasure

ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))


def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    trainset = datasets.MRIDatasetNumpySlices(train_dir)
    testset = datasets.MRIDatasetNumpySlices(test_dir)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config_train.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config_train.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


def train(model, trainloader, optimizer, epochs, filename=None):
    print(f"Training (on device: {config_train.DEVICE})...\n")

    for epoch in range(epochs):
        print("EPOCH: ", epoch + 1)
        running_loss, total_ssim = 0.0, 0.0

        n_batches = len(trainloader)

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

            running_loss += loss.item()
            total_ssim += ssim(predictions, targets).item()

            if index % config_train.LOSS_PRINT_FREQ == config_train.LOSS_PRINT_FREQ - 1:
                plot_title = f"epoch: {epoch} batch: {index}"
                utils.plot_predicted_batch(images, targets, predictions, title=plot_title, show=True)

                print(f'batch {(index + 1)} out of {n_batches}\t'
                      f'loss: {running_loss / config_train.LOSS_PRINT_FREQ:.3f} '
                      f'ssim {total_ssim / config_train.LOSS_PRINT_FREQ:.3f}')

                running_loss = 0.0
                total_ssim = 0.0

        print("\nTime for this epoch: ", time.time() - start)
        print()

    print("End of this round.")

    if filename is not None:
        model.save(config_train.TRAINED_MODEL_CLIENT_DIR, filename)


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
