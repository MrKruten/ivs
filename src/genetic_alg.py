import copy
import numpy as np
import pickle

from modelErrors import calculate_mse
from mamdani import compute_rule_activations, activate_conclusions, defuzzify_center_of_gravity_single_point
from helpers import create_file_inside_of_dir


class GeneticAlg:
    def __init__(self, rules, X_train, y_train, population_size: int, elite_size: int, max_epochs: int, log_path: str):
        self.rules = rules
        self.X_train = X_train
        self.y_train = y_train
        self.population_size = population_size
        self.elite_size = elite_size
        self.max_epochs = max_epochs
        self.best_loss = float('inf')
        self.loss_history = []
        self._antecedent_params_count = self._calculate_antecedent_params_counts()
        self.txt_file_path = log_path + "gen.txt"
        self.rules_file_path = log_path + "gen.pkl"
        create_file_inside_of_dir(log_path, "gen.txt")
        create_file_inside_of_dir(log_path, "gen.pkl")

    def _calculate_antecedent_params_counts(self):
        counts = []
        for rule in self.rules:
            count = sum(len(antecedent.parameters) for antecedent in rule.antecedents)
            counts.append(count)
        return counts

    def train(self):
        population = self._init_population()
        for epoch in range(self.max_epochs):
            fitness_scores = self._calc_fitness(population)
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            elite_population = [population[i] for i in elite_indices]
            min_scores = min(fitness_scores)
            if min_scores < self.best_loss:
                self.best_loss = min_scores
                self._update_rules_params(elite_population[0])

            new_population = elite_population
            while len(new_population) < self.population_size:
                parent_1, parent_2 = self._select_parents(population, fitness_scores)
                child = self._crossover(parent_1, parent_2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            self.loss_history.append(min_scores)
            text = f"Эпоха {epoch + 1}/{self.max_epochs}, best fitness score: {self.best_loss}"
            print(text)
            with open(self.txt_file_path, 'a+') as f:
                f.write(text + '\n')
            with open(self.rules_file_path, 'wb') as file:
                pickle.dump(self.rules, file)

        return self.rules, self.loss_history

    def _update_rules_params(self, params):
        antecedent_index = 0
        for rule in self.rules:
            for antecedent in rule.antecedents:
                antecedent.parameters = params[antecedent_index]
                antecedent_index += 1

    def _init_population(self):
        return [self._init_individual() for _ in range(self.population_size)]

    def _init_individual(self):
        antecedent_params = []
        for rule in self.rules:
            for antecedent in rule.antecedents:
                min_values = min(antecedent.parameters)
                max_values = max(antecedent.parameters)
                antecedent_params.append(
                    [np.random.uniform(min_values, max_values) for _ in range(len(antecedent.parameters))])
        return antecedent_params

    def _calc_fitness(self, population):
        fitness_scores = []
        for params in population:
            cloned_rules = [copy.deepcopy(rule) for rule in self.rules]
            antecedent_params_index = 0
            for rule in cloned_rules:
                for antecedent in rule.antecedents:
                    antecedent.parameters = params[antecedent_params_index]
                    antecedent_params_index += 1

            defuzzified_values = []
            for inputs in self.X_train:
                activations, consequent_memberships, consequent_centers = compute_rule_activations(cloned_rules,
                                                                                                   inputs)
                activated_conclusions = activate_conclusions(activations, consequent_memberships)
                defuzzified_values.append(
                    defuzzify_center_of_gravity_single_point(activated_conclusions, consequent_centers))
            loss = calculate_mse(self.y_train, defuzzified_values)
            text = f"ошибка - {loss}"
            print(text)
            with open(self.txt_file_path, 'a+') as file:
                file.write(text + '\n')
            fitness_scores.append(1 / (1 + loss))
        return fitness_scores

    def _select_parents(self, population, fitness_scores):
        parent_1 = self._select_parent(population, fitness_scores)
        parent_2 = self._select_parent(population, fitness_scores)
        return parent_1, parent_2

    @staticmethod
    def _select_parent(population, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_point = np.random.uniform(0, total_fitness)
        accumulated_fitness = 0
        for i, fitness in enumerate(fitness_scores):
            accumulated_fitness += fitness
            if accumulated_fitness >= selection_point:
                return population[i]

    @staticmethod
    def _crossover(parent_1, parent_2):
        crossover_point = np.random.randint(0, len(parent_1))
        child = parent_1[:crossover_point] + parent_2[crossover_point:]
        return child

    @staticmethod
    def _mutate(params):
        mutation_rate = 0.1
        for antecedent_params in params:
            for i in range(len(antecedent_params)):
                if np.random.rand() < mutation_rate:
                    antecedent_params[i] += np.random.normal(scale=0.1)
        return params