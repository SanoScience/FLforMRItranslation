import datetime
from os import path
import torch
from configs.enums import *


"""
    This is the configuration file. 
    Not all variables are used in every run. It depends on how the server and clients are launched and which aggregation method is used.
"""


# Variable set to true for testing locally
# It affects i.a. filepaths construction, server address.
# Also determines if multiple workers are used by DataLoader (see PyTorch dataloader)
LOCAL = False

# model parameters
NORMALIZATION = NormalizationType.BN
N_GROUP_NORM = 32


# client parameters
METRICS = ["loss", "ssim", "pnsr", "mse", "masked_mse", "relative_error"]
N_EPOCHS_CLIENT = 4
CLIENT_SAVING_FREQ = 10  # how often (round-wise) the model is saved

T1_TO_T2 = True  # False means the translation will be done from T2 to T1
LOSS_TYPE = LossFunctions.MSE_DSSIM
BATCH_SIZE = 32
IMAGE_SIZE = (240, 240)
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 8
BATCH_PRINT_FREQ = 10

# USED ONLY: when the server and clients are started saperatly
# port address
PORT = "8084"
# training parameters
CLIENT_TYPE = ClientTypes.FED_MRI
AGGREGATION_METHOD = AggregationMethods.FED_MEAN


# federated learning parameters
N_ROUNDS = 4
SAVING_FREQUENCY = 2
MIN_FIT_CLIENTS = MIN_AVAILABLE_CLIENTS = 5
FRACTION_FIT = 1.0

# SPECIALIZED METHOD
# FedOpt
TAU = 1e-3
# FedProx
PROXIMAL_MU = 0.01
STRAGGLERS = 0.2
# FedAvgM
MOMENTUM = 0.9

# centralized train
N_EPOCHS_CENTRALIZED = 50

# directories
if LOCAL:
    DATA_ROOT_DIR = path.join(path.expanduser("~"), "data")
else:
    DATA_ROOT_DIR = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL"

EVAL_DATA_DIRS = [path.join(DATA_ROOT_DIR, "lgg_26", "test"),
                  path.join(DATA_ROOT_DIR, "hgg_26", "test"),
                  path.join(DATA_ROOT_DIR, "wu_minn_26", "test"),
                  path.join(DATA_ROOT_DIR, "hcp_mgh_masks", "test"),
                  path.join(DATA_ROOT_DIR, "oasis_26", "test")]

now = datetime.datetime.now()
CENTRALIZED_DIR = f"{DATA_ROOT_DIR}/trained_models/model-mgh-centralized-{LOSS_TYPE.name}-ep{N_EPOCHS_CENTRALIZED}-lr{LEARNING_RATE}-{NORMALIZATION.name}-{now.date()}-{now.hour}h"
_REPRESENTATIVE_WORD = CLIENT_TYPE if CLIENT_TYPE == ClientTypes.FED_BN else AGGREGATION_METHOD
TRAINED_MODEL_SERVER_DIR = f"{DATA_ROOT_DIR}/trained_models/model-{_REPRESENTATIVE_WORD.name}-{LOSS_TYPE.name}-lr{LEARNING_RATE}-rd{N_ROUNDS}-ep{N_EPOCHS_CLIENT}-{NORMALIZATION.name}-{now.date()}"
