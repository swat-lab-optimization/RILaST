from abc import ABC, abstractmethod
from typing import Dict, List, Union
from rilast.validators.abstract_validator import AbstractValidator
from rilast.generators.abstract_generator import AbstractGenerator
from rilast.common.road_validity_check import is_valid_road



class RoadValidator(AbstractValidator):
    def __init__(self, generator: AbstractGenerator, map_size: int):
        self.map_size = map_size
        super().__init__(generator=generator)

    
    def is_valid(self, test: List[float]) -> (bool, str):
        valid, invalid_info = is_valid_road(test, self.map_size)
        return valid, invalid_info

    # You can add non-abstract methods here if needed