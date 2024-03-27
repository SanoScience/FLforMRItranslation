import os
import sys
import pickle
import torch
import importlib

from configs import config_train
from src import datasets, models, loss_functions, visualization
from torch.utils.data import DataLoader


class DifferentConfigs(Exception):
    pass


def import_from_filepath(filepath):
    module_name = filepath.replace('/', '_').replace('.py', '')  # Create a valid module name
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    # model and testset path from the command line    
    test_dir = sys.argv[1]
    model_path = sys.argv[2]
    BATCH_SIZE = int(sys.argv[3])
    model_dir = '/'.join(e for e in model_path.split('/')[:-1])
    representative_test_dir = test_dir.split('/')[-2]

    # verifying if the translation is the same direction as the trained model 
    config_path = os.path.join(os.path.dirname(model_path), "config.py")
    try:
        imported_config = import_from_filepath(config_path)
    
        if imported_config.TRANSLATION != config_train.TRANSLATION:
            raise DifferentConfigs(f"Different direction of translation. In for the trained model TRANSLATION={imported_config.TRANSLATION}")
    except FileNotFoundError:
        print(f"WARNING: The config file not found at {config_path}. The direction of the translation not verified!")

    testset = datasets.MRIDatasetNumpySlices([test_dir], translation_direction=config_train.TRANSLATION)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    
    if "prox" in model_path.lower():
        mu = imported_config.PROXIMAL_MU
        criterion = loss_functions.LossWithProximalTerm(proximal_mu=mu, base_loss_fn=loss_functions.DssimMse()) 
    else:
        criterion = loss_functions.DssimMse()
   
    unet = models.UNet(criterion).to(config_train.DEVICE)

    try:
        unet.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        FileNotFoundError(f"You are in {os.getcwd()} and there is no given path")
        exit()

    images, targets = next(iter(testloader))

    images = images.to(config_train.DEVICE)
    predictions = unet(images)

    metrics = unet.evaluate(testloader, with_masked_ssim=True, save_preds_dir=os.path.join(model_dir, "preds", representative_test_dir))
  
    filepath = os.path.join(model_dir, f"test_{representative_test_dir}_ssiim_{metrics['val_ssim']:.2f}.pkl")

    with open(filepath, "wb") as file:
        pickle.dump(metrics, file)
    print(f"Saved to : {filepath}")
