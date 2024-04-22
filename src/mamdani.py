from rule import Rule


def compute_rule_activations(rules: list[Rule],
                             inputs: list[float]) -> tuple[list[float], list[list[float]], list[list[float]]]:
    activations = []
    consequent_memberships = []  # Степени принадлежности для активизации заключений
    consequent_centers = []  # Центры термов в консеквентах
    for i, rule in enumerate(rules):
        antecedent_values = [antecedent.fuzzy_value(inputs[j]) for j, antecedent in enumerate(rule.antecedents)]
        rule_activation = min(antecedent_values)
        activations.append(rule_activation)
        # Получаем степени принадлежности для каждого элементарного подзаключения
        consequent_memberships.append([rule.consequent.fuzzy_value(inputs[j]) for j in range(len(inputs))])
        consequent_centers.append(rule.consequent.parameters)
    return activations, consequent_memberships, consequent_centers


def activate_conclusions(activations: list[float], consequent_memberships: list[list[float]],
                         rule_weights: list[float] = None) -> list[list[float]]:
    if rule_weights is None:
        rule_weights = [1] * len(activations)  # По умолчанию веса равны 1
    activated_conclusions = []
    for activation, memberships, weight in zip(activations, consequent_memberships, rule_weights):
        # Активируем заключение путем взятия минимального значения из степени принадлежности,
        # умноженного на степень активации правила и на весовой коэффициент
        activated_conclusion = [min(membership, activation) * weight for membership in memberships]
        activated_conclusions.append(activated_conclusion)
    return activated_conclusions


def defuzzify_center_of_gravity_single_point(activated_conclusions: list[list[float]],
                                             consequent_parameters: list[list[float]]) -> float:
    total_weighted_value = 0
    total_membership = 0

    for activated_conclusion, parameters in zip(activated_conclusions, consequent_parameters):
        for conclusion, param in zip(activated_conclusion, parameters):
            total_weighted_value += conclusion * param
            total_membership += conclusion

    if total_membership != 0:
        center_of_gravity = total_weighted_value / total_membership
    else:
        center_of_gravity = 0

    return center_of_gravity

