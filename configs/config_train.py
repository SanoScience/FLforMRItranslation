import datetime
from os import path
import torch
from configs.enums import *

LOCAL = False

N_EPOCHS_CENTRALIZED = 32

# training parameters
CLIENT_TYPE = ClientTypes.FED_AVG
LOSS_TYPE = LossFunctions.MSE_DSSIM
AGGREGATION_METHOD = AggregationMethods.FED_AVGM


# model parameters
NORMALIZATION = NormalizationType.BN
N_GROUP_NORM = 32


# client parameters
METRICS = ["loss", "ssim", "zoomed_ssim", "pnsr", "mse"]
N_EPOCHS_CLIENT = 3
BATCH_SIZE = 32
IMAGE_SIZE = (240, 240)
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 8
BATCH_PRINT_FREQ = 10

# addresses
PORT = "8084"

# federated learning parameters
N_ROUNDS = 20
SAVING_FREQUENCY = 1
MIN_FIT_CLIENTS = 5
FRACTION_FIT = 1.0
MIN_AVAILABLE_CLIENTS = 5
# FedOpt
TAU = 1e-3
# FedProx
PROXIMAL_MU = 0.5
STRAGGLERS = 0.2
# FedAvgM
MOMENTUM = 0.9

# directories
if LOCAL:
    DATA_ROOT_DIR = os.path.join(os.path.expanduser("~"), "data")
else:
    DATA_ROOT_DIR = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL"

EVAL_DATA_DIRS = [path.join(DATA_ROOT_DIR, "lgg_26", "test"),
                  path.join(DATA_ROOT_DIR, "hgg_26", "test"),
                  path.join(DATA_ROOT_DIR, "wu_minn_26", "test"),
                  path.join(DATA_ROOT_DIR, "hcp_mgh_masks", "test"),
                  path.join(DATA_ROOT_DIR, "oasis_26", "test")]

now = datetime.datetime.now()
CENTRALIZED_DIR = f"{DATA_ROOT_DIR}/trained_models/model-centralized-{LOSS_TYPE.name}-ep{N_EPOCHS_CENTRALIZED}-lr{LEARNING_RATE}-{NORMALIZATION.name}-{now.date()}-{now.hour}h"
_REPRESENTATIVE_WORD = CLIENT_TYPE if CLIENT_TYPE == ClientTypes.FED_BN else AGGREGATION_METHOD
TRAINED_MODEL_SERVER_DIR = f"{DATA_ROOT_DIR}/trained_models/model-{_REPRESENTATIVE_WORD.name}-{LOSS_TYPE.name}-lr{LEARNING_RATE}-rd{N_ROUNDS}-ep{N_EPOCHS_CLIENT}-{NORMALIZATION.name}-{now.date()}-{now.hour}h"
