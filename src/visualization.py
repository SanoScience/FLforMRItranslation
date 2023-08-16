from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_learning_curves(loss_histories, labels, title=None, ylabel="Loss", xlabel="Rounds"):
    plt.figure(figsize=(10, 6))

    for loss_values, label in zip(loss_histories, labels):
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, label=label)

    if title is not None:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.show()


# visualization
def plot_hist(tensor: Tensor, bins=240, title: str = None):
    plt.hist(tensor.detach().numpy().ravel(), bins=bins)
    plt.title(title)
    plt.show()


def plot_batch(to_plot: List[torch.Tensor], show=True, filepath: str = None, title="", cmap="gray"):
    batch_size = len(to_plot[0])
    fig, axs = plt.subplots(len(to_plot), batch_size, figsize=(3 * batch_size, 15))

    fig.suptitle(title)

    for i in range(batch_size):
        for j, set_to_plot in enumerate(to_plot):
            img_max = torch.max(set_to_plot)
            img_min = torch.min(set_to_plot)

            axs[j, i].imshow(set_to_plot[i].numpy()[0], cmap=cmap, vmin=0.0, vmax=1.0)
            axs[j, i].set_title(f'input\nmin: {img_min:.2f} max: {img_max:.2f}')
            axs[j, i].axis('off')

    if filepath is not None:
        plt.savefig(filepath)
        fig.clf()
        plt.close()
    if show:
        plt.show()
