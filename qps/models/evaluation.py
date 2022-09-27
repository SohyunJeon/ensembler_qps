import numpy as np
from sklearn.metrics import mean_squared_error
import copy


def cal_r2(y, y_pred):
    if y.ndim!=1:
        y = y.ravel()
    if y_pred.ndim != 1:
        y_pred = y_pred.ravel()
    corr_val = np.corrcoef(y, y_pred)[0, 1]
    r2 = corr_val ** 2
    return r2


def cal_rmse(y, y_pred):
    if y.ndim!=1:
        y = y.ravel()
    if y_pred.ndim != 1:
        y_pred = y_pred.ravel()
    return np.sqrt(mean_squared_error(y, y_pred))



def cal_residual(y, y_pred):
    return abs(y - y_pred)


