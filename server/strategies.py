import os
import torch
import logging

from common import config_train, utils

import flwr as fl
from flwr.server.criterion import ClientProxy
from flwr.common import Scalar, FitRes, Parameters

from typing import List, Tuple, Dict, Union, Optional
from collections import OrderedDict


class SaveModelStrategy(fl.server.strategy.FedAvg):
    # NOW VERIFY IF IT WORKS
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict)

            directory = config_train.TRAINED_MODEL_SERVER_DIR
            utils.try_create_dir(directory)

            torch.save(self.model.state_dict(), f"{directory}/round{server_round}.pth")
            print(f"Saved round {server_round} aggregated parameters to {directory}")

        return aggregated_parameters, aggregated_metrics


