from modelErrors import calculate_mse
from mamdani import compute_rule_activations, activate_conclusions, defuzzify_center_of_gravity_single_point


def train_consequents(X_train, y_train, rules, initial_learning_rate=0.1, num_epochs=1000,
                      patience=5, decay_rate=0.9):
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
                if learning_rate < 1e-2:
                    print("Скорость обучения слишком мала. Прекращение обучения.")
                    break

        print(f'Epoch {epoch}, Loss: {total_loss}, Learning Rate: {learning_rate}')
        losses.append(total_loss)

    return consequent_centers, losses
