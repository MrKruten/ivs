import pickle

from modelErrors import calculate_mse
from mamdani import compute_rule_activations, activate_conclusions, defuzzify_center_of_gravity_single_point
from helpers import create_file_inside_of_dir


def train_consequents(X_train, y_train, rules, log_path, initial_learning_rate=0.1, num_epochs=1000,
                      patience=5, decay_rate=0.9):
    txt_file_path = log_path + "grad.txt"
    rules_file_path = log_path + "grad.pkl"
    create_file_inside_of_dir(log_path, "grad.txt")
    create_file_inside_of_dir(log_path, "grad.pkl")
    learning_rate = initial_learning_rate
    best_loss = float('inf')
    bad_epochs = 0
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0

        for i, inputs in enumerate(X_train):
            activations, consequent_memberships, consequent_centers = compute_rule_activations(rules, inputs)
            activated_conclusions = activate_conclusions(activations, consequent_memberships)
            defuzzified_values = defuzzify_center_of_gravity_single_point(activated_conclusions, consequent_centers)
            if not isinstance(defuzzified_values, list):
                defuzzified_values = [defuzzified_values]
            # Вычисление градиента функции потерь по параметрам консеквентов
            gradients = []
            for j, consequent_param in enumerate(consequent_centers):
                gradient = []
                for k, defuzzified_value in enumerate(defuzzified_values):
                    if j < len(activated_conclusions) and k < len(activated_conclusions[j]) and \
                            activated_conclusions[j][k] != 0:
                        gradient.append(-2 * (y_train[i] - defuzzified_value) * activated_conclusions[j][k])
                    else:
                        gradient.append(0)
                gradients.append(gradient)

            # Обновление параметров консеквентов с использованием градиентного спуска
            for j, consequent_param in enumerate(consequent_centers):
                for k, param in enumerate(consequent_param):
                    if k < len(gradients[j]):
                        consequent_centers[j][k] -= learning_rate * gradients[j][k]

            total_loss += calculate_mse(y_train, defuzzified_values)

        if total_loss < best_loss:
            best_loss = total_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                learning_rate *= decay_rate
                if learning_rate < 1e-4:
                    text = "Скорость обучения слишком мала. Прекращение обучения."
                    print(text)
                    with open(txt_file_path, 'a+') as file:
                        file.write(text + '\n')
                    break

        text = f'Эпоха {epoch+1}, ошибка: {total_loss}, скорость обучения: {learning_rate}'
        print(text)
        with open(txt_file_path, 'a+') as file:
            file.write(text + '\n')
        losses.append(total_loss)
        with open(rules_file_path, 'wb') as file:
            pickle.dump(rules, file)

    return consequent_centers, losses
