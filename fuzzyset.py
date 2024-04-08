from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import os

from fuzzyterm import FuzzyTerm


@dataclass(init=False)
class FuzzySet:
    def __init__(self, name: str, terms: list[FuzzyTerm], feature: list[float]):
        self.name = name
        self.term_mapping: dict[str, FuzzyTerm] = {term.name: term for term in terms}
        self.feature = feature
        self.Y: list[float] = []

    def __str__(self):
        return self.name

    # Метод для построения графиков функций принадлежности термов
    def build_terms_graph(self, filename):
        for term in self.term_mapping:
            X = np.arange(min(self.feature), max(self.feature), 0.01)
            Y = []
            for x in X:
                Y.append(self.term_mapping[term].fuzzy_value(x))
            self.Y.append(Y)
            plt.plot(X, Y, label=self.term_mapping[term].name)

        plt.ylabel('Принадлежность')
        plt.xlabel('Значение параметра')
        plt.title(self.name)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
