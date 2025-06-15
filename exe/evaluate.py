"""Model evaluation script for federated learning results.
Loads trained models and evaluates them on test datasets."""

import os
import sys
import pickle
import torch
import importlib

from configs import config_train, enums
from src.ml import custom_metrics, datasets, models
from src.utils import visualization
from torch.utils.data import DataLoader



class DifferentTranslationError(Exception):
    pass


def import_from_filepath(to_import_filepath):
    """Dynamically import Python module from filepath."""
    # Create valid module name by replacing invalid characters
    module_name = to_import_filepath.replace('/', '_').replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, to_import_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    # Default configuration
    target_dir = None  # Optional target directory
    TRANSLATION = config_train.TRANSLATION  # Translation direction from config
    wanted_metrics = config_train.METRICS  # Metrics to compute
    input_target_union = False  # Whether to use intersection of input/target
    squeeze = False  # Whether to squeeze output dimensions
    BATCH_SIZE = 1  # Batch size for evaluation

    # Set paths based on environment
    if config_train.LOCAL:
        # Local testing paths
        test_dir = "C:\\Users\\JanFiszer\\data\\mri\\zoomed_ssim_test"
        model_path = "/trained_models/fl/flair_to_t2/model-fedadam-MSE_DSSIM-FLAIRT2-lr0.001-rd32-ep4-2024-04-07/round32.pth"
        representative_test_dir = "oasis"
    else:
        # Production paths from command line
        test_dir = sys.argv[1]  # Directory containing test data
        model_path = sys.argv[2]  # Path to model weights
        # BATCH_SIZE = int(sys.argv[3])
        # evaluate to have a
        if len(sys.argv) > 5:
            target_dir = sys.argv[5]

        representative_test_dir = test_dir.split(os.path.sep)[-2]

    # Load configuration file
    if len(sys.argv) > 4:
        config_path = sys.argv[4]
    else:
        config_path = os.path.join(os.path.dirname(model_path), "config.py")

    model_dir = '/'.join(e for e in model_path.split(os.path.sep)[:-1])

    print("Model dir is: ", model_dir)

    # Check if dataset has tumor mask for zoomed SSIM metric
    if representative_test_dir in ["ucsf_150", "hgg_125", "lgg"]:
        masks_available = True
    else:
        masks_available = False

    try:
        # verifying if the translation is the same direction as the trained model
        imported_config = import_from_filepath(config_path)
        TRANSLATION = imported_config.TRANSLATION
        print(f"\n\nTranslations: {imported_config.TRANSLATION}")

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
        
    print(f"Testing on the data from: {test_dir}")

    # Load data for evaluation
    testset = datasets.MRIDatasetNumpySlices(test_dir,
                                             target_dir=target_dir,
                                             translation_direction=TRANSLATION,
                                             squeeze=squeeze,
                                             metric_mask_dir=metric_mask_dir,  # Only use mask if available for dataset
                                             input_target_set_union=input_target_union)

    # Create dataloader with batch size 1 for detailed metrics
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    # Initialize appropriate loss function based on model type
    if "prox" in model_path.lower():
        mu = imported_config.PROXIMAL_MU
        criterion = custom_metrics.LossWithProximalTerm(proximal_mu=mu, base_loss_fn=custom_metrics.DssimMseLoss())
    else:
        criterion = custom_metrics.DssimMseLoss()

    print(f"Taken criterion is: {criterion}")
    
    # Initialize model with selected criterion
    unet = models.UNet(criterion).to(config_train.DEVICE)

    # Load model weights, exit if file not found
    try:
        print(f"Loading model from: {model_path}")
        unet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"You are in {os.getcwd()} and there is no given path")
        exit()

    print(f"Model and data loaded; evaluation starts...")

    # Setup directories for saving predictions and evaluation results
    save_preds_dir = os.path.join(model_dir, "../preds", representative_test_dir)
    eval_path = os.path.join(model_dir, "eval", representative_test_dir)

    # Perform evaluation and compute metrics with standard deviations
    metrics, stds = unet.evaluate(testloader,
                                  wanted_metrics=wanted_metrics,
                                  save_preds_dir=save_preds_dir,  # Save model predictions
                                  plots_path=eval_path,  # Save evaluation plots
                                  compute_std=True,  # Calculate standard deviations
                                  plot_metrics_distribution=True,  # Generate distribution plots
                                  low_ssim_value=0.4)  # Threshold for SSIM warnings

    # Save computed metrics and standard deviations
    metric_filepath = os.path.join(model_dir, f"metrics_{representative_test_dir}_ssim_{metrics['val_ssim']:.2f}.pkl")
    std_filepath = os.path.join(model_dir, f"std_{representative_test_dir}_ssim_{metrics['val_ssim']:.2f}.pkl")

    with open(metric_filepath, "wb") as file:
        pickle.dump(metrics, file)
    print(f"Metrics saved to : {metric_filepath}")

    with open(std_filepath, "wb") as file:
        pickle.dump(stds, file)
    print(f"Standard deviations saved to : {std_filepath}")

