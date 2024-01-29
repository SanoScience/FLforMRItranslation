# from skimage.metrics import structural_similarity as ssim
# import numpy as np
#
# import os
# import sys
# import pickle
# import torch
#
# from configs import config_train
# from src import datasets, models, loss_functions, visualization
# from torch.utils.data import DataLoader
# if __name__ == '__main__':
#     print(sys.argv)
#     test_dir = sys.argv[1]
#     model_path = sys.argv[2]
#     BATCH_SIZE = int(sys.argv[3])
#
#     testset = datasets.MRIDatasetNumpySlices([test_dir])
#     testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
#
#     # unet = models.UNet().to(config_train.DEVICE)
#     criterion = loss_functions.DssimMse()
#     unet = models.UNet(criterion).to(config_train.DEVICE)
#
#     with torch.no_grad():
#         for images_cpu, targets_cpu in testloader:
#             images = images_cpu.to(config_train.DEVICE)
#             targets = targets_cpu.to(config_train.DEVICE)
#
#             predictions = unet(images)
#
#             masks =
#             masks_bool = np.array(masks, dtype=bool)
#             masks_bool = np.invert(masks_bool)
#             ssim = ssim(out, fa, masks_bool)
#
#             for metric_name, metrics_obj in metrics.items():
#                 if isinstance(metrics_obj, loss_functions.LossWithProximalTerm):
#                     metrics_obj = metrics_obj.base_loss_fn
#                 metric_value = metrics_obj(predictions, targets)
#                 metrics_values[metric_name] += metric_value.item()
#
#             n_steps += 1

