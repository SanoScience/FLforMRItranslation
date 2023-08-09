import datetime
import os
import torch
from configs.enums import *

LOCAL = True

N_EPOCHS_CENTRALIZED = 12

# training parameters
CLIENT_TYPE = ClientTypes.FED_PROX
LOSS_TYPE = LossFunctions.PROX
AGGREGATION_METHOD = AggregationMethods.FED_PROX

# BATCH_NORMALIZATION = True if CLIENT_TYPE == ClientTypes.FED_BN else False
N_EPOCHS_CLIENT = 1
BATCH_SIZE = 10
N_GROUP_NORM = 16
IMAGE_SIZE = (240, 240)
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 8
BATCH_PRINT_FREQ = 10

# addresses
PORT = "8087"

# federated learning
N_ROUNDS = 5
SAVING_FREQUENCY = 8
TAU = 0.001
MIN_FIT_CLIENTS = 2
FRACTION_FIT = 1.0
MIN_AVAILABLE_CLIENTS = 2
PROXIMAL_MU = 0.5
STRAGGLERS = 0.6

# directories
if LOCAL:
    DATA_ROOT_DIR = os.path.join(os.path.expanduser("~"), "data")
else:
    DATA_ROOT_DIR = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL"

now = datetime.datetime.now()
# TRAINED_MODEL_CLIENT_DIR = f"./trained_models/client/model-lr{LEARNING_RATE}-ep{N_EPOCHS_CLIENT}-{now.date()}-{now.hour}_{now.minute}"
_REPRESENTATIVE_WORD = CLIENT_TYPE if CLIENT_TYPE == ClientTypes.FED_BN else AGGREGATION_METHOD
TRAINED_MODEL_SERVER_DIR = f"./trained_models/server/model-{_REPRESENTATIVE_WORD.name}-lr{LEARNING_RATE}-rd{N_ROUNDS}-ep{N_EPOCHS_CLIENT}-{now.date()}-{now.hour}h"
