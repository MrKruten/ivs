def trapezoidal_set(x: float, parameters: list[float]):
    a, b, c, d = parameters
    return max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0)


# Функция для получения параметров трапециевидных функций принадлежности
def get_trapezoid_params(count_terms, maximum_of_terms, minimum_of_terms):
    step = (maximum_of_terms - minimum_of_terms) / (count_terms + count_terms - 1)
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
