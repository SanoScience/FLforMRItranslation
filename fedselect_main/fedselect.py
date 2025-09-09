# Importing Libraries
import copy
import os.path
import pickle

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, OrderedDict, Tuple, Optional, Any

# Custom Libraries
from fedselect_main.utils.options import lth_args_parser
from fedselect_main.utils.train_utils import prepare_dataloaders, get_data
from fedselect_main.pflopt.optimizers import local_alt, MaskLocalAltSGD
from fedselect_main.lottery_ticket import init_mask_zeros, delta_update, init_mask
from fedselect_main.broadcast import (
    broadcast_server_to_client_initialization,
    div_server_weights,
    add_masks,
    add_server_weights,
)
import random
from torchvision.models import resnet18

from src.ml.models import UNet
from configs import config_train, enums
from src.ml import custom_metrics, datasets

from src.fl.clients import load_data
def evaluate(
    model: nn.Module, ldr_test: torch.utils.data.DataLoader, args: Any
) -> float:
    """Evaluate model accuracy on test data loader.

    Args:
        model: Neural network model to evaluate
        ldr_test: Test data loader
        args: Arguments containing device info

    Returns:
        float: Average accuracy on test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_accuracy = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ldr_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
            average_accuracy += acc
        average_accuracy /= len(ldr_test)
    return average_accuracy


def train_personalized(
    model: nn.Module,
    ldr_train: torch.utils.data.DataLoader,
    mask: OrderedDict,
    args: Any,
    initialization: Optional[OrderedDict] = None,
    verbose: bool = True,
    eval: bool = True,
) -> Tuple[nn.Module, float]:
    """Train model with personalized local alternating optimization.

    Args:
        model: Neural network model to train
        ldr_train: Training data loader
        mask: Binary mask for parameters
        args: Training arguments
        initialization: Optional initial model state
        verbose: Whether to print training progress
        eval: Whether to evaluate during training

    Returns:
        Tuple containing:
            - Trained model
            - Final training loss
    """
    model.train()

    if initialization is not None:
        model.load_state_dict(initialization)
    optimizer = custom_metrics.MaskedAdam(model.parameters(), mask, lr=config_train.LEARNING_RATE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config_train.LEARNING_RATE)
    # optimizer = MaskLocalAltSGD(model.parameters(), mask, lr=0.01)

    print(f"Optimizer: {optimizer}")

    epochs = args.la_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = custom_metrics.DssimMseLoss()
    train_loss = 0
    # with tqdm(total=epochs) as pbar:
    for i in range(epochs):
        train_loss = local_alt(
            model,
            criterion,
            optimizer,
            ldr_train,
            device,
            clip_grad_norm=args.clipgradnorm,
        )
        if verbose:
            print(f"\t\tEpoch: {i} \tLoss: {train_loss}")
        # pbar.update(1)
            # pbar.set_postfix({"Loss": train_loss})
    return model, train_loss


def fedselect_algorithm(
    model: UNet,
    args: Any,
    idxs_users: List[str],
    # dataset_train: torch.utils.data.Dataset,
    # dataset_test: torch.utils.data.Dataset,
    # dict_users_train: Dict[int, np.ndarray],
    # dict_users_test: Dict[int, np.ndarray],
    # labels: np.ndarray,
):
    """Main FedSelect federated learning algorithm.

    Args:
        model: Neural network model
        args: Training arguments
        dataset_train: Training dataset
        dataset_test: Test dataset
        dict_users_train: Mapping of users to training data indices
        dict_users_test: Mapping of users to test data indices
        labels: Data labels
        idxs_users: List of user indices

    Returns:
        Dict containing:
            - client_accuracies: Accuracy history for each client
            - labels: Data labels
            - client_masks: Final client masks
            - args: Training arguments
            - cross_client_acc: Cross-client accuracy matrix
            - lth_convergence: Lottery ticket convergence history
    """
    # initialize model dir
    drd = config_train.ROOT_DIR
    lt = config_train.LOSS_TYPE.name
    t = f"{config_train.TRANSLATION[0].name}{config_train.TRANSLATION[1].name}"
    lr = config_train.LEARNING_RATE
    rd = config_train.N_ROUNDS
    ec = config_train.N_EPOCHS_CLIENT
    n = config_train.NORMALIZATION.name
    d = config_train.now.date()
    h = config_train.now.hour

    model_dir = f"{drd}/trained_models/model-fedselect-{lt}-{t}-lr{lr}-rd{rd}-ep{ec}-{d}"
    os.makedirs(model_dir, exist_ok=True)

    print(f"All the results of the experiment will be stored in {model_dir}")
    # initialize model
    initial_state_dict = copy.deepcopy(model.state_dict())
    com_rounds = args.com_rounds

    data_dir = args.data_dir
    # initialize dataset
    train_ds, test_ds = {}, {}
    for client_idx in idxs_users:
        # load all the dataloader for each client
        train_ds[client_idx], test_ds[client_idx], _ = load_data(os.path.join(data_dir, client_idx),
                                                                                        config_train.BATCH_SIZE,
                                                                                        config_train.NUM_WORKERS)
    # initialize server
    client_accuracies = [{i: 0 for i in idxs_users} for _ in range(com_rounds)]
    client_histories = {i: {f"val_{metric_name}": [] for metric_name in config_train.METRICS} for i in idxs_users}
    client_state_dicts = {i: copy.deepcopy(initial_state_dict) for i in idxs_users}
    client_state_dict_prev = {i: copy.deepcopy(initial_state_dict) for i in idxs_users}
    client_masks = {i: None for i in idxs_users}
    client_masks_prev = {i: init_mask_zeros(model) for i in idxs_users}
    server_accumulate_mask = OrderedDict()
    server_weights = OrderedDict()
    lth_iters = args.lth_epoch_iters
    prune_rate = args.prune_percent / 100
    prune_target = args.prune_target / 100
    lottery_ticket_convergence = []

    # Begin FL
    for round_num in range(com_rounds):
        print(f"ROUND {round_num}")
        round_loss = 0
        for current_client in idxs_users:
            # client specific directory
            client_dir = os.path.join(model_dir, current_client)
            # initialize model
            model.load_state_dict(client_state_dicts[current_client])
            # get data
            ldr_train = train_ds[current_client]
            print(f"Client {current_client} evaluation")
            print(f"Evaluation after aggregation...")

            metrics = model.evaluate(testloader=test_ds[current_client], plots_path=client_dir, plot_filename=f"b4-local-train-round-{round_num}")

            for metric_name, metric_value in metrics.items():
                client_histories[current_client][metric_name].append(metric_value)

            # Update LTN_i on local data
            client_mask = client_masks_prev.get(current_client)

            print("\tTraining")
            # Update u_i parameters on local data
            # 0s are global parameters, 1s are local parameters
            client_model, loss = train_personalized(model, ldr_train, client_mask, args)
            round_loss += loss

            print(f"Local evaluation...")
            client_model.evaluate(testloader=test_ds[current_client], plots_path=client_dir, plot_filename=f"after-local-train-round-{round_num}")

            # Send u_i update to server
            if round_num < com_rounds - 1:
                server_accumulate_mask = add_masks(server_accumulate_mask, client_mask)
                server_weights = add_server_weights(
                    server_weights, client_model.state_dict(), client_mask
                )
            else:  # last round saving all the results
                print(f"Saving the model and history for client {current_client}")
                # pickle dump - history
                with open(os.path.join(client_dir, "history.pkl"), 'wb') as f:
                    pickle.dump(client_histories[current_client], f)
                # saving the personalized model
                torch.save(client_state_dicts[current_client], os.path.join(client_dir, "model.pth"))

            client_state_dicts[current_client] = copy.deepcopy(client_model.state_dict())
            client_masks[current_client] = copy.deepcopy(client_mask)

            if round_num % lth_iters == 0 and round_num != 0:
                client_mask = delta_update(
                    prune_rate,
                    client_state_dicts[current_client],
                    client_state_dict_prev[current_client],
                    client_masks_prev[current_client],
                    bound=prune_target,
                    invert=True,
                )
                client_state_dict_prev[current_client] = copy.deepcopy(client_state_dicts[current_client])
                client_masks_prev[current_client] = copy.deepcopy(client_mask)

        round_loss /= len(idxs_users)
        print(f"This round loss: {round_loss}")
        # cross_client_acc = cross_client_eval(
        #     model,
        #     client_state_dicts,
        #     dataset_train,
        #     dataset_test,
        #     dict_users_train,
        #     dict_users_test,
        #     args,
        # )

        # accs = torch.diag(cross_client_acc)
        # for i in range(len(accs)):
        #     client_accuracies[round_num][i] = accs[i]
        # print("Client Accs: ", accs, " | Mean: ", accs.mean())

        if round_num < com_rounds - 1:
            # Server averages u_i
            server_weights = div_server_weights(server_weights, server_accumulate_mask)
            # Server broadcasts non lottery ticket parameters u_i to every device
            for current_client in idxs_users:
                client_state_dicts[current_client] = broadcast_server_to_client_initialization(
                    server_weights, client_masks[current_client], client_state_dicts[current_client]
                )
            server_accumulate_mask = OrderedDict()
            server_weights = OrderedDict()

    print(client_histories)
    # cross_client_acc = cross_client_eval(
    #     model,
    #     client_state_dicts,
    #     dataset_train,
    #     dataset_test,
    #     dict_users_train,
    #     dict_users_test,
    #     args,
    #     no_cross=False,
    # )

    # out_dict = {
    #     "client_accuracies": client_accuracies,
    #     "labels": labels,
    #     "client_masks": client_masks,
    #     "args": args,
    #     "cross_client_acc": cross_client_acc,
    #     "lth_convergence": lottery_ticket_convergence,
    # }
    #
    # return out_dict


def cross_client_eval(
    model: nn.Module,
    client_state_dicts: Dict[int, OrderedDict],
    dataset_train: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset,
    dict_users_train: Dict[int, np.ndarray],
    dict_users_test: Dict[int, np.ndarray],
    args: Any,
    no_cross: bool = True,
) -> torch.Tensor:
    """Evaluate models across clients.

    Args:
        model: Neural network model
        client_state_dicts: Client model states
        dataset_train: Training dataset
        dataset_test: Test dataset
        dict_users_train: Mapping of users to training data indices
        dict_users_test: Mapping of users to test data indices
        args: Evaluation arguments
        no_cross: Whether to only evaluate on own data

    Returns:
        torch.Tensor: Matrix of cross-client accuracies
    """
    cross_client_acc_matrix = torch.zeros(
        (len(client_state_dicts), len(client_state_dicts))
    )
    idx_users = list(client_state_dicts.keys())
    for _i, i in enumerate(idx_users):
        model.load_state_dict(client_state_dicts[i])
        for _j, j in enumerate(idx_users):
            if no_cross:
                if i != j:
                    continue
            # eval model i on data from client j
            _, ldr_test = prepare_dataloaders(
                dataset_train,
                dict_users_train[j],
                dataset_test,
                dict_users_test[j],
                args,
            )
            acc = evaluate(model, ldr_test, args)
            cross_client_acc_matrix[_i, _j] = acc
    return cross_client_acc_matrix


def get_cross_correlation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Get cross correlation between two tensors using F.conv2d.

    Args:
        A: First tensor
        B: Second tensor

    Returns:
        torch.Tensor: Cross correlation result
    """
    # Normalize A
    A = A.cuda() if torch.cuda.is_available() else A
    B = B.cuda() if torch.cuda.is_available() else B
    A = A.unsqueeze(0).unsqueeze(0)
    B = B.unsqueeze(0).unsqueeze(0)
    A = A / (A.max() - A.min()) if A.max() - A.min() != 0 else A
    B = B / (B.max() - B.min()) if B.max() - B.min() != 0 else B
    return F.conv2d(A, B)


def run_base_experiment(model: nn.Module, args: Any) -> None:
    """Run base federated learning experiment.

    Args:
        model: Neural network model
        args: Experiment arguments
    """
    # dataset_train, dataset_test, dict_users_train, dict_users_test, labels = get_data(
    #     args
    # )
    # idxs_users = np.arange(args.num_users * args.frac)
    # m = max(int(args.frac * args.num_users), 1)
    # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    # idxs_users = [int(i) for i in idxs_users]
    if config_train.LOCAL:
        idxs_users = ["mega_small_wu_minn", "mega_small_hgg"]
    else:
        if  enums.ImageModality.FLAIR in config_train.TRANSLATION:
            idxs_users = ["hgg_125", "oasis", "lgg", "ucsf_150"]
        else:
            # idxs_users = ["hgg_125", "oasis", "lgg", "ucsf_150", "hcp_wu_minn", "hcp_mgh_masks"]
            idxs_users = ["lgg", "hcp_mgh_masks"]


    print(f"Selected clients: {idxs_users}")

    fedselect_algorithm(
        model,
        args,
        idxs_users
    )


def load_model(args: Any) -> nn.Module:
    """Load and initialize model.

    Args:
        args: Model arguments

    Returns:
        nn.Module: Initialized model
    """
    device = config_train.DEVICE
    args.device = device
    model = resnet18(pretrained=args.pretrained_init)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    model = model.to(device)
    return model.to(device)


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # Argument Parser
    args = lth_args_parser()

    # Set the seed
    setup_seed(args.seed)
    # model = load_model(args)

    # Initialize model components
    criterion = custom_metrics.DssimMseLoss()  # Combined SSIM and MSE loss
    unet = UNet(criterion).to(config_train.DEVICE)  # Move model to GPU if available
    mask = custom_metrics.init_mask(unet)

    run_base_experiment(unet, args)
