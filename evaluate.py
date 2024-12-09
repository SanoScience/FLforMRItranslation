import os
import sys
import pickle
import torch
import importlib
import random

from configs import config_train, enums
from src.ml import custom_metrics, datasets, models
from src.utils import visualization
from torch.utils.data import DataLoader

import numpy as np


class DifferentTranslationError(Exception):
    pass


def import_from_filepath(to_import_filepath):
    module_name = to_import_filepath.replace('/', '_').replace('.py', '')  # Create a valid module name
    spec = importlib.util.spec_from_file_location(module_name, to_import_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    # base values
    target_dir = None
    segmentation_task = False
    TRANSLATION = config_train.TRANSLATION
    wanted_metrics = config_train.METRICS
    input_target_union = False
    squeeze = False
    BATCH_SIZE = 1

    if config_train.LOCAL:
        test_dir = "C:\\Users\\JanFiszer\\data\\mri\\zoomed_ssim_test"
        model_path = "C:\\Users\\JanFiszer\\repos\\FLforMRItranslation\\trained_models\\fl\\flair_to_t2\\model-fedadam-MSE_DSSIM-FLAIRT2-lr0.001-rd32-ep4-2024-04-07\\round32.pth"
        representative_test_dir = "oasis"
    else:
        test_dir = sys.argv[1]
        model_path = sys.argv[2]
        # BATCH_SIZE = int(sys.argv[3])
        # evaluate to have a
        if len(sys.argv) > 5:
            target_dir = sys.argv[5]

        representative_test_dir = test_dir.split(os.path.sep)[-2]

    if len(sys.argv) > 4:
        config_path = sys.argv[4]
    else:
        config_path = os.path.join(os.path.dirname(model_path), "config.py")

    model_dir = '/'.join(e for e in model_path.split(os.path.sep)[:-1])

    print("Model dir is: ", model_dir)

    if representative_test_dir in ["ucsf_150", "hgg_125", "lgg"]:
        masks_available = True
    else:
        masks_available = False

    try:
        # verifying if the translation is the same direction as the trained model
        imported_config = import_from_filepath(config_path)
        TRANSLATION = imported_config.TRANSLATION
        print(f"\n\nTranslations: {imported_config.TRANSLATION}")

        segmentation_task = imported_config.TRANSLATION[1] == enums.ImageModality.MASK or imported_config.TRANSLATION[1] == enums.ImageModality.TUMOR
        if segmentation_task:
            print("\nMask as the target modality, evaluation for segmenatation task\n")

        if imported_config.TRANSLATION[1] == enums.ImageModality.TUMOR:
            input_target_union = True
            print("Taking `tumor` datasets, only union will be taken.")

    except FileNotFoundError:
        print(f"WARNING: The config file not found at {config_path}. The direction of the translation not verified!")

    if masks_available:
        metric_mask_dir = "mask"
    else:
        metric_mask_dir = None
        wanted_metrics.remove("zoomed_ssim")

    testset = datasets.MRIDatasetNumpySlices(test_dir,
                                             target_dir=target_dir,
                                             translation_direction=TRANSLATION,
                                             binarize=segmentation_task,
                                             squeeze=squeeze,
                                             metric_mask_dir=metric_mask_dir,
                                             input_target_set_union=input_target_union)

    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    if "prox" in model_path.lower():
        mu = imported_config.PROXIMAL_MU
        criterion = custom_metrics.LossWithProximalTerm(proximal_mu=mu, base_loss_fn=custom_metrics.DssimMse())
    elif segmentation_task:
        criterion = custom_metrics.BinaryDiceLoss()
    else:
        criterion = custom_metrics.DssimMse()

    print(f"Taken criterion is: {criterion}")
    unet = models.UNet(criterion).to(config_train.DEVICE)

    try:
        print(f"Loading model from: {model_path}")
        unet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"You are in {os.getcwd()} and there is no given path")
        exit()

    print(f"Testing on the data from: {test_dir}")

    if BATCH_SIZE > 1:
        if testset.metric_mask:
            images, targets, _ = next(iter(testloader))
        else:
            images, targets = next(iter(testloader))

        images = images.to(config_train.DEVICE)
        predictions = unet(images)

        visualization.plot_batch([images.to('cpu'), targets.to('cpu'), predictions.to('cpu').detach()],
                                 filepath=os.path.join(test_dir, f"segmenation_results"))

    print(f"Model and data loaded; evaluation starts...")

    save_preds_dir = os.path.join(model_dir, "preds", representative_test_dir)

    metrics, stds = unet.evaluate(testloader,
                                  wanted_metrics=wanted_metrics,
                                  save_preds_dir=save_preds_dir,
                                  compute_std=True)

    metric_filepath = os.path.join(model_dir, f"metrics_{representative_test_dir}_ssim_{metrics['val_ssim']:.2f}.pkl")
    std_filepath = os.path.join(model_dir, f"std_{representative_test_dir}_ssim_{metrics['val_ssim']:.2f}.pkl")

    with open(metric_filepath, "wb") as file:
        pickle.dump(metrics, file)
    print(f"Metrics saved to : {metric_filepath}")

    with open(std_filepath, "wb") as file:
        pickle.dump(stds, file)
    print(f"Standard deviations saved to : {std_filepath}")

