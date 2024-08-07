import os
import sys
import pickle
import torch
import importlib

from configs import config_train, enums
from src.ml import custom_metrics, datasets, models
from src.utils import visualization
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice


if __name__ == '__main__':
    dice = Dice()
    target_dir = sys.argv[1]
    predicted_dir = sys.argv[2]

    eval_dataset = datasets.VoluminEvaluation(target_dir, predicted_dir)
    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    dice_scores = []

    for batch_index, batch in enumerate(dataloader):
        targets_cpu, predicted_cpu = batch[0], batch[1]

        targets = targets_cpu.to(config_train.DEVICE)
        predicted = predicted_cpu.to(config_train.DEVICE)

        dice_scores.append(dice(targets, predicted))

    average_dice = sum(dice_scores)/len(dice_scores)
    print(f"The 3D dice score is: {average_dice}")

