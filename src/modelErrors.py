import numpy as np


def calculate_mape(true_values, predicted_values):
    non_zero_values = [(true, pred) for true, pred in zip(true_values, predicted_values) if true != 0]
    n = len(non_zero_values)
    mape = sum(abs((true - pred) / true) for true, pred in non_zero_values if true > 1e-6) / n * 100
    return mape


def calculate_mse(true_values, predicted_values):
    # Преобразуем true_values и predicted_values в списки, если они не являются итерируемыми объектами
    true_values_list = [true_values] if not hasattr(true_values, '__iter__') else true_values
    predicted_values_list = [predicted_values] if not hasattr(predicted_values, '__iter__') else predicted_values

    # Вычисляем квадрат ошибки для каждой пары значений
    squared_errors = [(true - pred) ** 2 for true, pred in zip(true_values_list, predicted_values_list)]

    # Исключаем проблемные значения (например, nan) из списка квадратов ошибок
    squared_errors = [error for error in squared_errors if not np.isnan(error)]

    # Проверяем, что остались значения для вычисления среднего значения
    if squared_errors:
        mse = np.mean(squared_errors)
    else:
        # Если список пустой, возвращаем nan
        mse = np.nan
    return mse


def calculate_rmse(true_values, predicted_values):
    mse = calculate_mse(true_values, predicted_values)
    rmse = np.sqrt(mse)
    return rmse


def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)