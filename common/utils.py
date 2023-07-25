import logging
import os
import traceback

import matplotlib.pyplot as plt
import torch
from torch import Tensor


class MinMaxScalar(object):
    def __init__(self, desired_range=(0.0, 1.0)):
        super(MinMaxScalar, self).__init__()
        self.desired_range = desired_range

    def __call__(self, tensor: Tensor):
        # TODO: implement variable desired_range usage
        max_value = torch.max(tensor).data
        min_value = torch.min(tensor)

        # converting to float to be able to perform tensor multiplication
        # otherwise an error
        return (tensor / max_value).float()

    def __repr__(self):
        return "Min-max scaler"


# visualization
def plot_hist(tensor: Tensor, bins=240, title=None):
    plt.hist(tensor.detach().numpy().ravel(), bins=bins)
    plt.title(title)
    plt.show()


def plot_predicted_batch(images, target, predictions, show=True, filepath=None, title="", cmap="gray"):
    batch_size = len(images)
    fig, axs = plt.subplots(3, batch_size, figsize=(3 * batch_size, 20))

    plt.title(title)

    for i in range(batch_size):
        axs[0, i].imshow(images[i].numpy()[0], cmap=cmap)
        axs[1, i].imshow(target[i].numpy()[0], cmap=cmap)
        axs[2, i].imshow(predictions[i].detach().numpy()[0], cmap=cmap)

        axs[0, i].set_title('input')
        axs[1, i].set_title('target')
        axs[2, i].set_title('predicted')

        axs[0, i].axis('off')
        axs[1, i].axis('off')
        axs[2, i].axis('off')

    if filepath is not None:
        plt.savefig(filepath)
    if show:
        plt.show()


def try_create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        logging.warning(f"Directory {dir_name} already exists. You may overwrite your files or create some collisions!")

    except FileNotFoundError:
        ex = FileNotFoundError(f"The path {dir_name} to directory willing to be created doesn't exist. You are in {os.getcwd()}.")

        traceback.print_exception(FileNotFoundError, ex, ex.__traceback__)
