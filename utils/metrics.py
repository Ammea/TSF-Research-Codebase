import numpy as np
from sklearn.metrics import r2_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr

def get_metrics(y_true, y_pred, only_nonzero=False):
    if only_nonzero:
        non_index = np.nonzero(y_true)[0]
        y_pred, y_true = y_pred[non_index], y_true[non_index]
    n = len(y_true)
    n_non_zero = sum(y_true > 0)
    gap = y_pred - y_true
    gap_abs = abs(gap)
    gap_abs_per = np.array(list(map(lambda x: x[0] / x[1] if x[1] > 0 else 0, zip(gap_abs, y_true))))
    min_max = np.array(list(map(lambda x: min(x[0], x[1]) / max(x[0], x[1]) if max(x[0], x[1]) > 0 else 0, zip(y_pred, y_true))))


    metric_dict = {}
    metric_dict['1-wmape'] = f'{(1 - gap_abs.sum() / y_true.sum()):.6f}'
    metric_dict['1-mape'] = f'{(1 - gap_abs_per.sum() / n_non_zero):.6f}'
    metric_dict['r2'] = f'{r2_score(y_true, y_pred):.6f}'
    metric_dict['mean_minmax'] = f'{(min_max.sum() / n_non_zero):.6f}'
    metric_dict['num_obs'] = (n_non_zero)
    metric_dict['all_obs'] = n
    # metric_dict['mae'] = f'{gap_abs.sum() / n:.4f}'

    return pd.DataFrame([list(metric_dict.values())], columns=list(metric_dict.keys())), metric_dict
