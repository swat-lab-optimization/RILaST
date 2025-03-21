import numpy as np
from abc import ABC, abstractmethod
from rilast.common.road_validity_check import is_valid_road
from rilast.generators.abstract_generator import AbstractGenerator

class AbstractValidator(ABC):
    def __init__(self, generator: AbstractGenerator):
        self.generator = generator

    @abstractmethod
    def is_valid(self, test) -> (bool, str):
        pass

        

    # You can add non-abstract methods here if needed