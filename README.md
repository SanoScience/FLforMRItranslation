# Federated learning for MRI image-to-image translation

<img src="./imgs/cover-ismrm.png">

## Brief
This project implements classical (centralized) and federated learning approaches for MRI image translation between different modalities (T1 -> T2, FLAIR -> T2, etc.). It includes several federated learning strategies like FedAvg, FedProx, FedBN, and a novel one - FedBAdam. All the details are described in the article: [Validation of ten federated learning strategies for multi-contrast image-to-image MRI data synthesis from heterogeneous sources](https://www.biorxiv.org/content/10.1101/2025.02.09.637305v1.full-text). The work has been also presented in Singapore on the [ISMRM](https://www.ismrm.org/24m/) conference as [Federated image-to-image MRI translation from heterogeneous multiple-sites data](https://archive.ismrm.org/2024/2221.html)


## Abstract 
Deep learning (DL)-based image synthesis has recently gained enormous interest in medical imaging, allowing for generating multi-contrast data and therefore, the recovery of missing samples from interrupted or artefact-distorted acquisitions. However, the accuracy of DL models heavily relies on the representativeness of the training datasets naturally characterized by their distributions, experimental setups or preprocessing schemes. These complicate generalizing DL models across multi-site heterogeneous data sets while maintaining the confidentiality of the data. One of the possible solutions is to employ federated learning (FL), which enables the collaborative training of a DL model in a decentralized manner, demanding the involved sites to share only the characteristics of the models without transferring their sensitive medical data. The paper presents a DL-based magnetic resonance (MR) data translation in a FL way. We introduce a new aggregation strategy called FedBAdam that couples two state-of-the-art methods with complementary strengths by incorporating momentum in the aggregation scheme and skipping the batch normalization layers. The work comprehensively validates 10 FL-based strategies for an image-to-image multi-contrast MR translation, considering healthy and tumorous brain scans from five different institutions. Our study has revealed that the FedBAdam shows superior results in terms of mean squared error and structural similarity index over personalized methods, like the FedMRI, and standard FL-based aggregation techniques, such as the FedAvg or FedProx, considering multi-site multi-vendor heterogeneous environment. The FedBAdam has prevented the overfitting of the model and gradually reached the optimal model parameters, exhibiting no oscillations.

## Installation
The project requires Python 3.8+ and PyTorch 1.8+. All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Configuration
The main configuration file is configs/config_train.py. Key settings include:

```
# Environment
LOCAL = False  # Set True for local testing
NODE_FILENAME = "SERVERNODE.txt"

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_EPOCHS_CLIENT = 4  # epochs per round for FL clients
N_EPOCHS_CENTRALIZED = 10  # epochs for classical training

# Federated Learning Parameters
N_ROUNDS = 32
MIN_FIT_CLIENTS = MIN_AVAILABLE_CLIENTS = 4
FRACTION_FIT = 1.0

# Translation Direction
TRANSLATION = (ImageModality.T1, ImageModality.T2)  # e.g., T1->T2 translation
```


## Running Training
Classical (Centralized) Training

```python exe/trainings/classical_train.py <data_directory>```

Federated Learning
1. Start the server:
```
python exe/trainings/run_server.py <port_number> <strategy_name>
# Example: python exe/trainings/run_server.py 8080 fedavg
```
2. Initialize clients (to run multiple times)
```
python exe/trainings/run_client_train.py <data_directory> <client_id> <server_address> <strategy_name>
# Example: python exe/trainings/run_client_train.py /data/client1 1 localhost:8080 fedavg
```

Using Bash Scripts
The scripts/ directory contains helper scripts for running experiments:
```
# Start server with multiple strategies
./scripts/run_server.sh

# Start multiple clients
./scripts/run_clients.sh

# Run evaluation on test sets
./scripts/evaluate_models.sh
```
