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


def plot_batch(to_plot, show=True, filepath=None, title="", cmap="gray"):
    batch_size = len(to_plot[0])
    fig, axs = plt.subplots(len(to_plot), batch_size, figsize=(3 * batch_size, 15))

    fig.suptitle(title)

    for i in range(batch_size):
        for j, set_to_plot in enumerate(to_plot):
            img_max = torch.max(set_to_plot)
            img_min = torch.min(set_to_plot)

            # targets_max = torch.max(targets[i])
            # targets_min = torch.min(targets[i])
            #
            # predicted_max = torch.max(predictions[i])
            # predicted_min = torch.min(predictions[i])

            axs[j, i].imshow(set_to_plot[i].numpy()[0], cmap=cmap)
            # axs[1, i].imshow(targets[i].numpy()[0], cmap=cmap)
            # axs[2, i].imshow(predictions[i].detach().numpy()[0], cmap=cmap, vmin=0.0, vmax=1.0)

            axs[j, i].set_title(f'input\nmin: {img_min:.2f} max: {img_max:.2f}')
            # axs[1, i].set_title(f'target\nmin: {targets_min:.2f} max: {targets_max:.2f}')
            # axs[2, i].set_title(f'predicted\nmin: {predicted_min:.2f} max: {predicted_max:.2f}')

            axs[j, i].axis('off')
            # axs[1, i].axis('off')
            # axs[2, i].axis('off')

    if filepath is not None:
        plt.savefig(filepath)
        fig.clf()
        plt.close()
    if show:
        plt.show()


# def plot_batch(images, target, show=True, filepath=None, title="", cmap="gray"):
#     batch_size = len(images)
#     fig, axs = plt.subplots(2, batch_size, figsize=(3 * batch_size, 8))
#
#
#     for i in range(batch_size):
#         axs[0, i].imshow(images[i].numpy()[0], cmap=cmap)
#         axs[1, i].imshow(target[i].numpy()[0], cmap=cmap)
#
#         axs[0, i].set_title('input')
#         axs[1, i].set_title('target')
#
#         axs[0, i].axis('off')
#         axs[1, i].axis('off')
#
#     if filepath is not None:
#         plt.savefig(filepath)
#         fig.clf()
#         plt.close()
#     if show:
#         plt.show()


