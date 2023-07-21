import os
import torch

from common import config_train, models

import flwr as fl
from flwr.server.criterion import ClientProxy
from flwr.common import Scalar, FitRes, Parameters

from typing import List, Tuple, Dict, Union, Optional
from collections import OrderedDict

unet = models.UNet


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated parameters")

            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            params_dict = zip(unet.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            unet.load_state_dict(state_dict)

            directory = config_train.TRAINED_MODEL_DIR
            if not os.path.isdir(directory):
                os.mkdir(directory)

            torch.save(unet.state_dict(), f"{directory}/round{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


