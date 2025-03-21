import os
import numpy as np
import abc
import logging #as log
from abc import ABC, abstractmethod
from typing import Tuple, Dict
from rilast.validators.abstract_validator import AbstractValidator
from rilast.executors.abstract_executor import AbstractExecutor
from beamng_sim.code_pipeline.tests_generation import RoadTestFactory
from rilast.common.road_validity_check import min_radius
from rilast.common.vehicle_evaluate import evaluate_scenario

log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130
MIN_RADIUS_THRESHOLD = 47

class CurveExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, generator, test_validator: AbstractValidator= None, results_path: str = None):
        super().__init__(generator, test_validator, results_path)
        self._name = "CurveExecutor"
        self.min_fitness = 0.0125

    def _execute(self, test) -> float:

        min_curve = min_radius(test)

        if min_curve <= MIN_RADIUS_THRESHOLD:
            fitness = 0
        else:
            fitness = -1/min_curve
        
        return fitness