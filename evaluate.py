import os
import sys

import torch

from configs import config_train
from src import datasets, models, loss_functions, visualization
from torch.utils.data import DataLoader

if not config_train.LOCAL:
    os.chdir("repos/FLforMRItranslation")

if __name__ == '__main__':
    # test_dir = os.path.join(os.path.expanduser("~"), "data/hgg_transformed/validation")
    print(sys.argv)
    test_dir = sys.argv[1]
    model_path = sys.argv[2]
    BATCH_SIZE = int(sys.argv[3])

    testset = datasets.MRIDatasetNumpySlices([test_dir])
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    # unet = models.UNet().to(config_train.DEVICE)
    criterion = loss_functions.DssimMse()
    unet = models.UNet(criterion).to(config_train.DEVICE)
    if config_train.LOCAL:
        unet.load_state_dict(torch.load(os.path.join(model_path), map_location=torch.device('cpu')))
    else:
        try:
            unet.load_state_dict(torch.load(os.path.join(model_path)))
        except FileNotFoundError:
            FileNotFoundError(f"You are in {os.getcwd()} and there is no give path")

    images, targets = next(iter(testloader))

    images = images.to(config_train.DEVICE)
    predictions = unet(images)

    # loss, ssim = unet.test(testloader, criterion)
    # representative_test_dir = test_dir.split('/')[-2]
    # model_dir = '/'.join(e for e in model_path.split('/')[:-1])

    # filepath = os.path.join(model_dir, f"test_{representative_test_dir}_loss_{loss:.2f}_ssim{ssim:.2f}.jpg")
    # visualization.plot_batch([images.cpu(), targets.cpu(), predictions.cpu().detach()],
    #                          title=representative_test_dir,
    #                          show=False,
    #                          filepath=filepath)

    # print("Plot saved to :", filepath)
    # print(f"ssim {ssim} loss {loss}")
    # print("\nEvaluation ended.\n")

    metrics = unet.evaluate(testloader, criterion)
    metric_string = loss_functions.metrics_to_str(metrics, starting_symbol="")
    representative_test_dir = test_dir.split('/')[-2]
    model_dir = '/'.join(e for e in model_path.split('/')[:-1])

    filepath = os.path.join(model_dir, f"test_{representative_test_dir}_loss_{metrics['loss']:.2f}_ssim{metrics['ssim']:.2f}.jpg")
    visualization.plot_batch([images.cpu(), targets.cpu(), predictions.cpu().detach()],
                             title=representative_test_dir,
                             show=False,
                             filepath=filepath)

    print("Plot saved to :", filepath)
    print(metric_string)
    print("\nEvaluation ended.\n")
