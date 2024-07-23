import os
import sys
import pickle
import torch
import importlib

from configs import config_train, enums
from src.ml import custom_metrics, datasets, models
from torch.utils.data import DataLoader


class DifferentTranslationError(Exception):
    pass


def import_from_filepath(to_import_filepath):
    module_name = to_import_filepath.replace('/', '_').replace('.py', '')  # Create a valid module name
    spec = importlib.util.spec_from_file_location(module_name, to_import_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    # model and testset path from the command line    
    test_dir = sys.argv[1]
    model_path = sys.argv[2]
    BATCH_SIZE = int(sys.argv[3])

    if len(sys.argv) > 4:
        config_path = sys.argv[4]
    else:
        config_path = os.path.join(os.path.dirname(model_path), "config.py")

    model_dir = '/'.join(e for e in model_path.split('/')[:-1])
    representative_test_dir = test_dir.split('/')[-2]
    print("Model dir is: ", model_dir)
    # verifying if the translation is the same direction as the trained model 
    try:
        imported_config = import_from_filepath(config_path)
        
        if imported_config.TRANSLATION != config_train.TRANSLATION:
            raise DifferentTranslationError(f"Different direction of translation. In for the trained model TRANSLATION={imported_config.TRANSLATION}")
        else:
            print(f"Translations match: {imported_config.TRANSLATION}")

        segmenatation_task = imported_config.TRANSLATION[1] == enums.ImageModality.MASK
        if segmenatation_task:
            print("\nMask as the target modality, evaluation for segmenatation task\n")

    except FileNotFoundError:
        print(f"WARNING: The config file not found at {config_path}. The direction of the translation not verified!")
    
    testset = datasets.MRIDatasetNumpySlices([test_dir], translation_direction=config_train.TRANSLATION, binarize=segmenatation_task)
    testloader = DataLoader(testset, batch_size=1355, shuffle=False)
    if "prox" in model_path.lower():
        mu = imported_config.PROXIMAL_MU
        criterion = custom_metrics.LossWithProximalTerm(proximal_mu=mu, base_loss_fn=custom_metrics.DssimMse())
    elif segmenatation_task:
        criterion = custom_metrics.BinaryDiceLoss(binary_crossentropy=True)
    else:
        criterion = custom_metrics.DssimMse()
    
    print(f"Taken criterion is: {criterion}")
    unet = models.UNet(criterion).to(config_train.DEVICE)

    try:
        print(f"Loading model from: {model_path}")
        unet.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"You are in {os.getcwd()} and there is no given path")
        exit()
    
    print(f"Testing on the data from: {test_dir}")
    images, targets = next(iter(testloader))

    images = images.to(config_train.DEVICE)
    predictions = unet(images)

    print(f"Model and data loaed; evaluation starts...")
    if segmenatation_task:
        save_preds_dir = None
    else:
        save_preds_dir = os.path.join(model_dir, "preds", representative_test_dir)
        
    metrics = unet.evaluate(testloader, wanted_metrics=["jaccard", "dice", "loss"], save_preds_dir=save_preds_dir)
    
    if segmenatation_task:
        filepath = os.path.join(model_dir, f"test_{representative_test_dir}_jaccard_{metrics['val_jaccard']:.2f}.pkl")
    else:
        filepath = os.path.join(model_dir, f"test_{representative_test_dir}_ssim_{metrics['val_ssim']:.2f}.pkl")

    with open(filepath, "wb") as file:
        pickle.dump(metrics, file)
    print(f"Saved to : {filepath}")
