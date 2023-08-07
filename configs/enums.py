from enum import Enum


class ClientTypes(Enum):
    FED_AVG = 1
    FED_PROX = 2
    FED_BN = 3


class LossFunctions(Enum):
    MSE = 1
    MSE_DSSIM = 2
    PROX = 3
    RMSE_DDSIM = 4


class AggregationMethods(Enum):
    FED_AVG = 1
    FED_PROX = 2
    FED_ADAM = 3
    FED_ADAGRAD = 4
    FED_YOGI = 5
    FED_COSTW = 6
    FED_PID = 7
