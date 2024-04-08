import typing as t
from dataclasses import dataclass


@dataclass
class FuzzyTerm:
    name: str
    parameters: list[float]
    membership_function: t.Callable[[float, list[float]], float]

    def fuzzy_value(self, x: float) -> float:
        return self.membership_function(x, self.parameters)

    def __str__(self):
        return self.name

    def __getitem__(self, key):
        if key == self.name:
            return self
