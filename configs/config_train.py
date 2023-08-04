import datetime
import os
import torch
from configs.enums import *
# TODO: divide
# TODO: enum (?)

all_loss_functions = ["mse", "dssim", "with_prox"]

LOCAL = True

N_EPOCHS_CENTRALIZED = 12

# training parameters
CLIENT_TYPE = ClientTypes.FED_AVG
LOSS_TYPE = LossFunctions.MSE_DSSIM
AGGREGATION_METHOD = AggregationMetithods.FED_ADAM

N_EPOCHS_CLIENT = 1
BATCH_SIZE = 3
IMAGE_SIZE = (240, 240)
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 8
BATCH_PRINT_FREQ = 10

# addresses
PORT = "8087"

# federated learning
N_ROUNDS = 2
MIN_FIT_CLIENTS = 2
FRACTION_FIT = 1.0
MIN_AVAILABLE_CLIENTS = 2
PROXIMAL_MU = 0.5
STRAGGLERS = 0.6

# functions used while training
CRITERION = "with_prox"

# directories
if LOCAL:
    DATA_ROOT_DIR = os.path.join(os.path.expanduser("~"), "data")
else:
    DATA_ROOT_DIR = "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL"

now = datetime.datetime.now()
TRAINED_MODEL_CLIENT_DIR = f"./trained_models/client/model-lr{LEARNING_RATE}-ep{N_EPOCHS_CLIENT}-{now.date()}-{now.hour}_{now.minute}"
TRAINED_MODEL_SERVER_DIR = f"./trained_models/server/model-lr{LEARNING_RATE}-ep{N_EPOCHS_CLIENT}-{now.date()}-{now.hour}_{now.minute}"
