import socket

from common import config_train, models
from server.strategies import SaveModelStrategy

from typing import List, Tuple, Dict, Optional

import flwr as fl
from flwr.common import Metrics, NDArray, Scalar


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["ssim"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"ssim": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    unet = models.UNet()

    strategy = SaveModelStrategy(unet,
                                 evaluate_metrics_aggregation_fn=weighted_average,
                                 min_fit_clients=config_train.MIN_FIT_CLIENTS,
                                 min_available_clients=config_train.MIN_AVAILABLE_CLIENTS,
                                 fraction_fit=config_train.FRACTION_FIT)

    server_address = f"{socket.gethostname()}:{config_train.PORT}"

    fl.server.start_server(
        server_address=,
        config=fl.server.ServerConfig(num_rounds=config_train.N_ROUNDS),
        strategy=strategy
    )
