from typing import List

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import Tensor

import math

def plot_pred_tigth(to_plot, col_labels=None, img_size=None, cmap="gray", forecolor='black', ):

    fig=plt.figure()

    fig.patch.set_facecolor(forecolor)
    if forecolor == "black":
        textcolor = "white"
    else:
        textcolor = "black"
    dim_reduced_to_plot = []

    for images in to_plot:
        dim_reduced_to_plot.append([img[0, 0] for img in images])

    if img_size is None:
        # looking for a common min shape
        min_W = 10000
        min_H = 10000
        for img in dim_reduced_to_plot[0]:

            img_H, img_W = img.shape

            if min_W > img_W:
                min_W = img_W
            if min_H > img_H:
                min_H = img_H
    else:
        min_W, min_H = img_size


    def trim_image(image, min_width, min_height):
        img_H, img_W = image.shape

        to_trim_H = img_H - min_height
        to_trim_W = img_W - min_width

        result_img = image[math.ceil(to_trim_H/2): min_H + math.ceil(to_trim_H/2), math.ceil(to_trim_W/2): min_W+math.ceil(to_trim_W/2)]
        return result_img

    to_plot_trimmed = []
    for images in dim_reduced_to_plot:
        to_plot_trimmed.append([trim_image(img, min_W, min_H) for img in images])

    all_numpy_rows = [np.concatenate(images, axis=1) for images in to_plot_trimmed]
    total_numpy_array = np.concatenate(all_numpy_rows)

    plt.imshow(total_numpy_array, cmap=cmap)
    plt.axis("off")

    if col_labels:
        for index, col_label in enumerate(col_labels):
            plt.text(index * 240, 0, col_label, color=textcolor)


def plot_difference(target, predictions, row_labels=None, cmap="hsv", vmin=-0.5, vmax=0.5):

    fig, axs = plt.subplots(len(predictions), len(target), figsize=(3*len(target), 3*len(predictions)))

    for i in range(len(target)):
        for j, prediction in enumerate(predictions):
            # print(target[i].shape, prediction[i].shape)
            difference = target[i].numpy()[0][0] - prediction[i].numpy()[0][0]

            if row_labels:
                if i == 0:
                    axs[j, i].text(10, 150, row_labels[j], rotation="vertical")

            im = axs[j, i].imshow(difference, cmap, vmin=vmin, vmax=vmax)
            axs[j, i].axis("off")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)


def plot_learning_curves(loss_histories, labels, colors=None, linetypes=None, title=None, ylabel="Loss", xlabel="Rounds", ylim=None, figsize=None, legend=True):

    if figsize:
        plt.figure(figsize=figsize)

    if ylim:
        plt.ylim(ylim)

    linestyle = '-'

    for index, loss_values in enumerate(loss_histories):
        epochs = range(1, len(loss_values) + 1)

        label=labels[index]

        if linetypes:
            linestyle = linetypes[index]
        if colors:
            plt.plot(epochs, loss_values, label=label, linestyle=linestyle, color=colors[index])
        else:
            plt.plot(epochs, loss_values, label=label, linestyle=linestyle)

    if title is not None:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
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


def plot_pred(to_plot: List[torch.Tensor], col_labels, row_labels=None, show=True, filepath: str = None,
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
    if forecolor == "black":
        textcolor = "white"
    else:
        textcolor = "black"
    fig.tight_layout(pad=pad)
    fig.suptitle(title)

    for i, set_to_plot in enumerate(to_plot):
        for j in range(list_size):
            if not vertical:
                i, j = j, i

            img = set_to_plot[j][0].numpy()[0]

            axs[i, j].imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)

            if row_labels:
                if j == 0:
                    axs[i, j].text(10, 150, row_labels[i], color=textcolor, rotation="vertical")

            axs[i, j].set_title(title, color=textcolor)
            axs[i, j].axis("off")
            axs[i, j].autoscale_view('tight')

    for index, col_label in enumerate(col_labels):
        fig.text(index * 0.198 + 0.1, 1, col_label, color=textcolor)

    if filepath is not None:
        plt.savefig(filepath)
        fig.clf()
        plt.close()
    if show:
        plt.show()
