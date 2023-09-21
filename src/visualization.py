from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_learning_curves(loss_histories, labels, linetypes=None, title=None, ylabel="Loss", xlabel="Rounds", ylim=None):
    plt.figure(figsize=(10, 6))
    if ylim:
        plt.ylim(ylim)

    linestyle = '-'

    for index, loss_values in enumerate(loss_histories):
        epochs = range(1, len(loss_values) + 1)
        if linetypes:
            linestyle = linetypes[index]
        plt.plot(epochs, loss_values, label=labels[index], linestyle=linestyle)

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


def plot_batch(to_plot: List[List[torch.Tensor]], labels=None, show=True, filepath: str = None, title="", cmap="gray"):
    batch_size = len(to_plot[0])
    fig, axs = plt.subplots(batch_size, len(to_plot), figsize=(len(to_plot) * 4, 3 * batch_size))

    if labels is None:
        labels = ["Input", "Target", "Predicted"]

    fig.suptitle(title)

    for i in range(batch_size):
        for j, set_to_plot in enumerate(to_plot):
            img_max = torch.max(set_to_plot[i])
            img_min = torch.min(set_to_plot[i])

            axs[i, j].imshow(set_to_plot[i].numpy()[0], cmap=cmap, vmin=0.0, vmax=1.0)

            title = f'\nmin: {img_min:.2f} max: {img_max:.2f}'
            
            if i == 0:
                title = labels[j] + title

            axs[i, j].set_title(title)
            axs[i, j].axis("off")

    if filepath is not None:
        plt.savefig(filepath)
        fig.clf()
        plt.close()
    if show:
        plt.show()


def plot_pred(to_plot: List[torch.Tensor], labels, show=True, filepath: str = None,
              title="", cmap="gray", pad=0.5, forecolor='black', figsize=None, vertical=True):
    list_size = len(to_plot[0])
    if figsize is None:
        if vertical:
            figsize = (3 * list_size, len(to_plot) * 3)
        else:
            figsize = (len(to_plot) * 3, 3 * list_size)

    if vertical:
        fig, axs = plt.subplots(len(to_plot), list_size, figsize=figsize)
    else:
        fig, axs = plt.subplots(list_size, len(to_plot), figsize=figsize)
    fig.patch.set_facecolor(forecolor)

    fig.tight_layout(pad=pad)
    fig.suptitle(title)

    for i, set_to_plot in enumerate(to_plot):
        for j in range(list_size):
            if not vertical:
                i, j = j, i

            img_max = torch.max(set_to_plot[j][0])
            img_min = torch.min(set_to_plot[j][0])

            axs[i, j].imshow(set_to_plot[j][0].numpy()[0], cmap=cmap, vmin=0.0, vmax=1.0)

            title = f'\nmin: {img_min:.2f} max: {img_max:.2f}'
            if i == 0:
                title = labels[j] + title

            axs[i, j].set_title(title, color='white')
            axs[i, j].axis("off")

    if filepath is not None:
        plt.savefig(filepath)
        fig.clf()
        plt.close()
    if show:
        plt.show()
