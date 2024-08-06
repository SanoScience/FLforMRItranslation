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
    module_name = to_import_filepath.replace('/', '_').replace('.py', '')  # Create a valid module name
    spec = importlib.util.spec_from_file_location(module_name, to_import_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    # base values
    target_dir=None
    segmentation_task=False
    TRANSLATION = None

    if config_train.LOCAL:
        test_dir = "C:\\Users\\JanFiszer\\data\\mri\\zoomed_ssim_test"
        # test_dir = "C:\\Users\\JanFiszer\\data\\mri\\zoomed_ssim_test\\flair"
        target_dir = "C:\\Users\\JanFiszer\\data\\mri\\zoomed_ssim_test\\mask"
        model_path = "C:\\Users\\JanFiszer\\repos\\FLforMRItranslation\\trained_models\\centralized\\segmentation\\model-hgg_125-MSE_DSSIM-ep20-FLAIRMASK-lr0.0001-2024-08-01-16h\\best_model.pth"
    else:
        test_dir = sys.argv[1]
        model_path = sys.argv[2]
        BATCH_SIZE = int(sys.argv[3])

        if len(sys.argv) > 4:
            target_dir = sys.argv[4]
            
    if len(sys.argv) > 5:
        config_path = sys.argv[5]
    else:
        config_path = os.path.join(os.path.dirname(model_path), "config.py")

    model_dir = '/'.join(e for e in model_path.split(os.path.sep)[:-1])

    representative_test_dir = test_dir.split(os.path.sep)[-2]

    print("Model dir is: ", model_dir)
    # verifying if the translation is the same direction as the trained model 
    
    # print(test_dir)
    # print(model_dir)
    # print(target_dir)
    # print(representative_test_dir)
    try:
        imported_config = import_from_filepath(config_path)
        TRANSLATION = imported_config.TRANSLATION
        # if imported_config.TRANSLATION != TRANSLATION:
        #     raise DifferentTranslationError(f"Different direction of translation. In for the trained model TRANSLATION={imported_config.TRANSLATION}")
        # else:
        #     print(f"Translations match: {imported_config.TRANSLATION}")

        segmentation_task = imported_config.TRANSLATION[1] == enums.ImageModality.MASK or imported_config.TRANSLATION[1] == enums.ImageModality.TUMOR
        if segmentation_task:
            print("\nMask as the target modality, evaluation for segmenatation task\n")

    except FileNotFoundError:
        print(f"WARNING: The config file not found at {config_path}. The direction of the translation not verified!")
    
    if imported_config.TRANSLATION[1] == enums.ImageModality.TUMOR:
        print("Taking `tumor` datasets, only union will be taken.")

    # testset = datasets.MRIDatasetNumpySlices(test_dir, target_dir=target_dir, binarize=segmentation_task)
    testset = datasets.MRIDatasetNumpySlices(test_dir, 
                                             target_dir=target_dir,
                                             translation_direction=TRANSLATION,
                                             binarize=segmentation_task,
                                             squeeze=target_dir is not None,
                                             input_target_set_union=imported_config.TRANSLATION[1] == enums.ImageModality.TUMOR)
    
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

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

    print(f"Model and data loaed; evaluation starts...")
    if segmentation_task:
        save_preds_dir = None
    else:
        save_preds_dir = os.path.join(model_dir, "preds", representative_test_dir)
        
    metrics = unet.evaluate(testloader, wanted_metrics=["loss",  "dice_classification", "generalized_dice", "dice_2_class", "jaccard"], save_preds_dir=save_preds_dir)
    
    if segmentation_task:
        filepath = os.path.join(model_dir, f"test_{representative_test_dir}_dice_{metrics['val_dice_2_class']:.2f}.pkl")
    else:
        filepath = os.path.join(model_dir, f"test_{representative_test_dir}_zoomed_ssim_{metrics['val_zoomed_ssim']:.2f}.pkl")

    with open(filepath, "wb") as file:
        pickle.dump(metrics, file)
    print(f"Saved to : {filepath}")