import os
import torch
import torchvision
import datetime

from common.utils import MinMaxScalar
from client import loss_functions

# TODO: divide

N_EPOCHS_CLIENT = 64
BATCH_SIZE = 32
IMAGE_SIZE = (240, 240)
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 16

LOSS_PRINT_FREQ = 10

TRANSFORMS = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), MinMaxScalar()])
CRITERION = torch.nn.MSELoss()
OPTIMIZER = torch.optim.Adam

N_ROUNDS = 2
MIN_FIT_CLIENTS = 2
FRACTION_FIT = 0.75
MIN_AVAILABLE_CLIENTS = 2

# directories
DATA_DIR = os.path.join(os.path.expanduser("~"), "data\\with_val")

now = datetime.datetime.now()
TRAINED_MODEL_CLIENT_DIR = f"./trained_models/client/model-lr{LEARNING_RATE}-ep{N_EPOCHS_CLIENT}-{now.date()}-{now.hour}_{now.minute}"
TRAINED_MODEL_SERVER_DIR = f"./trained_models/server/model-lr{LEARNING_RATE}-ep{N_EPOCHS_CLIENT}-{now.date()}-{now.hour}_{now.minute}"
