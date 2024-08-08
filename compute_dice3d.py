import os
import sys
import pickle

from configs import config_train
from src.ml import custom_metrics, datasets
from src.utils import visualization
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice


if __name__ == '__main__':
    dice = Dice()
    target_dir = sys.argv[1]
    predicted_dir = sys.argv[2]

    eval_dataset = datasets.VolumeEvaluation(target_dir, predicted_dir)
    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    dice_scores = []
    false_positives = []

    for batch_index, batch in enumerate(dataloader):
        targets_cpu, predicted_cpu = batch[0], batch[1]

        targets = targets_cpu.to(config_train.DEVICE)
        predicted = predicted_cpu.to(config_train.DEVICE)

        dice_scores.append(dice(predicted.cpu().int(), targets.cpu()))
        false_positives.append(custom_metrics.false_positive_ratio(predicted.cpu().int(), targets.cpu().int()))

    average_dice = sum(dice_scores)/len(dice_scores)
    false_positive_ratio = sum(false_positives)/len(false_positives)

    print(f"The 3D dice score is: {average_dice}")
    print(f"The 3D false positive ratio is: {false_positive_ratio}")


    filepath = os.path.join(predicted_dir, f"test_dice3d_{average_dice:.2f}.pkl")

    with open(filepath, "wb") as file:
        pickle.dump({"dice3d": average_dice, "fpr": false_positive_ratio}, file)

