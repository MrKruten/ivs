import  numpy as np


def trapezoid_membership(x: float, parameters: list[float]):
    a, b, c, d = parameters
    return max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0)


def get_trapezoid_params(count_terms, maximum_of_terms, minimum_of_terms):
    step = (maximum_of_terms - minimum_of_terms) / (count_terms + count_terms - 1)
    print("trap step ", step)
    result = []
    pos = minimum_of_terms

    while len(result) < count_terms:
        a = pos - step
        b = pos
        c = pos + step
        d = pos + 2 * step
        result.append([a, b, c, d])
        pos += 2 * step
    return result


def gaussian_membership(x: float, parameters: list[float]) -> float:
    mean, std_dev = parameters
    return np.exp(-0.5 * ((x - mean) / std_dev) ** 2)


def bell_membership(x: float, parameters: list[float]) -> float:
    mean, std_dev = parameters
    return 1 / (1 + ((x - mean) / std_dev) ** 2)


def get_gaussian_params(count_terms, maximum_of_terms, minimum_of_terms):
    step = (maximum_of_terms - minimum_of_terms) / (count_terms - 1)
    print("guas step ", step)
    result = []
    pos = minimum_of_terms

    while len(result) < count_terms:
        mean = pos
        std_dev = step / 2  # Используем половину шага в качестве стандартного отклонения
        result.append([mean, std_dev])
        pos += step
    return result


def get_bell_params(count_terms, maximum_of_terms, minimum_of_terms):
    mean_step = (maximum_of_terms - minimum_of_terms) / (count_terms + count_terms - 1)
    print("trap step ", mean_step)
    result = []
    mean_pos = minimum_of_terms

    while len(result) < count_terms:
        mean = mean_pos
        std_dev = mean_step / 2  # Используем половину шага в качестве стандартного отклонения
        result.append([mean, std_dev])
        mean_pos += 2 * mean_step
    return result
