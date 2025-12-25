import numpy as np

def rmse(x, y):
    rmse = np.sqrt(np.mean((x - y) ** 2))

    return rmse

def lins_ccc(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = np.mean((x - x_mean) * (y - y_mean))
    x_var = np.var(x)
    y_var = np.var(y)

    ccc = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2 + 1e-8)

    return ccc

def pearson_corr(x, y):
    correlation_matrix = np.corrcoef(x, y)
    pearson_corr = correlation_matrix[0, 1]

    return pearson_corr
