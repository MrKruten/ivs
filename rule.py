from dataclasses import dataclass

from fuzzyterm import FuzzyTerm


@dataclass
class Rule:
    consequent: FuzzyTerm
    antecedents: list[FuzzyTerm]
    weight: float = None

    def __str__(self):
        text = 'ЕСЛИ '
        input_texts = [f'{input_el}' for input_el in self.antecedents]
        text += " И ".join(input_texts)
        text += f', ТО {self.consequent}.'
        text += f' Истина: {str(self.weight)}'
        return text
