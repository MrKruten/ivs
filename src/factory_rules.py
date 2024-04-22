from rule import Rule
from fuzzyterm import FuzzyTerm
from fuzzyset import FuzzySet
from memberships import trapezoid_membership, get_trapezoid_params, gaussian_membership, get_gaussian_params, get_bell_params, bell_membership


term_names = {
    1: ["Medium"],
    2: ["Low", "High"],
    3: ["Low", "Medium", "High"],
    4: ["Very Low", "Slightly low", "Slightly High", "Very High"],
    5: ["Very Low", "Slightly low", "Medium", "Slightly High", "Very High"],
    6: ["Very Low", "Low", "Slightly low", "Slightly High", "High", "Very High"],
    7: ["Very Low", "Low", "Slightly low", "Medium", "Slightly High", "High", "Very High"]
}


def create_rules(columns, target_column, target_column_name, number_terms, input_features, num_membership) -> list[Rule]:
    directory = f'./screens/l_{len(input_features)}_t_{number_terms}_m_{num_membership}/'
    fuzzy_sets = []

    if num_membership == 0:
        get_params = get_trapezoid_params
        membership = trapezoid_membership
    elif num_membership == 1:
        get_params = get_gaussian_params
        membership = gaussian_membership
    elif num_membership == 2:
        get_params = get_bell_params
        membership = bell_membership

    for index, column in enumerate(columns):
        ranges_of_parameters = get_params(number_terms, max(column), min(column))
        terms = []
        for num_term in range(number_terms):
            terms.append(FuzzyTerm(term_names[number_terms][num_term], ranges_of_parameters[num_term],
                                   membership))

        fuzzy_set = FuzzySet(
            input_features[index],
            terms=terms,
            feature=column
        )
        filename = f'{directory}{input_features[index]}.png'
        fuzzy_set.build_terms_graph(filename)
        fuzzy_sets.append(fuzzy_set)

    ranges_of_parameters = get_params(number_terms, max(target_column), min(target_column))
    terms = []
    for num_term in range(number_terms):
        terms.append(FuzzyTerm(term_names[number_terms][num_term], ranges_of_parameters[num_term],
                               membership))
    target_fuzzy_set = FuzzySet(target_column_name, terms=terms, feature=target_column)
    filename = f'{directory}{target_column_name}.png'
    target_fuzzy_set.build_terms_graph(filename)

    rules = create_rules_internal(fuzzy_sets, target_fuzzy_set)
    return rules


def create_rules_internal(fuzzy_sets, target_fuzzy_set, antecedents=[], index=0):
    rules = []
    if index == len(fuzzy_sets):
        for target_key, target_value in target_fuzzy_set.term_mapping.items():
            rules.append(Rule(antecedents=antecedents, consequent=target_value[target_key]))
        return rules
    else:
        for key, value in fuzzy_sets[index].term_mapping.items():
            rules.extend(create_rules_internal(fuzzy_sets, target_fuzzy_set, antecedents + [value[key]], index + 1))
        return rules
