import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import json
import os

from rule import Rule
from modelErrors import calculate_mse
from mamdani import compute_rule_activations, activate_conclusions, defuzzify_center_of_gravity_single_point
from factory_rules import create_rules
from genetic_alg import GeneticAlg
from gradient_descent import train_consequents


if __name__ == '__main__':
    if 'MY_CONFIG_PATH' in os.environ:
        config_path = os.environ['MY_CONFIG_PATH']
    else:
        config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    df = pd.read_csv(config['dataset_path'], low_memory=False)

    df = df.dropna()
    df = df.astype(float)

    if config['need_calc_correlation_matrix']:
        # Вычисление корреляции между признаками и целевым параметром
        correlation_matrix = df.corr()

        # Создание графика тепловой карты для визуализации корреляции
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

        # Вывод корреляции с целевым параметром в порядке убывания
        correlation_with_target = correlation_matrix[config['target_column']].sort_values(ascending=False)
        print(correlation_with_target)

    input_features = config['input_features']  # Список входных параметров
    output_feature = config['target_column']  # Выходной параметр, который мы хотим предсказать

    X = df[input_features]  # Создание DataFrame с входными параметрами
    y = df[output_feature]  # Создание Series с выходным параметром

    # Разделение данных на обучающую и тестовую выборки в соотношении 80/20,
    # с фиксированным random_state для воспроизводимости результатов
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'],
                                                        random_state=config['random_seed'])
    print("Размерность обучающей выборки:", X_train.shape)
    print("Размерность тестовой выборки:", X_test.shape)

    # Преобразование DataFrame в массивы numpy и затем в списки
    # Создание объекта MinMaxScaler
    scaler = MinMaxScaler()
    # Создание объекта MinMaxScaler для нормирования выходного параметра
    scaler_y = MinMaxScaler()

    # Нормирование выходного параметра
    y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1))

    # Нормирование признаков
    X_train_scaled = scaler.fit_transform(X_train)

    columns = []
    for i in range(len(input_features)):
        columns.append(X_train_scaled[:, i].tolist())

    y1 = y_train_scaled[:, 0].tolist()

    rules: list[Rule] = create_rules(columns, y1, output_feature, config['number_terms'], input_features, config['membership'])
    print("Количество правил: - ", len(rules))

    # Запуск без обучения
    if config['need_calc_without_training']:
        defuzzified_values = []

        for i in range(len(X_train)):
            inputs = [x[i] for x in columns]
            activations, consequent_memberships, consequent_centers = compute_rule_activations(rules, inputs)
            activated_conclusions = activate_conclusions(activations, consequent_memberships)
            defuzzified_value = defuzzify_center_of_gravity_single_point(activated_conclusions, consequent_centers)
            defuzzified_values.append(defuzzified_value)

        # y1 - это истинные значения, а defuzzified_values - это предсказанные значения после дефазификации.
        mse = calculate_mse(y1, defuzzified_values)
        print("\nMean Squared Error (MSE):", mse)

    try:
        print("start gradient descent")
        train_consequents(X_train=columns, y_train=y1, rules=rules, num_epochs=1000,
                          initial_learning_rate=0.1, log_path=config['log_path'])
        print("start genetic")
        gen_alg = GeneticAlg(rules=rules, X_train=columns, y_train=y1,
                             population_size=30, elite_size=5, max_epochs=1000, log_path=config['log_path'])
        best_antecedent_params, loss_history = gen_alg.train()
    except Exception as er:
        print('error', er)

    exit_word = ''
    while exit_word != 'exit':
        exit_word = input("Enter exit: ")
