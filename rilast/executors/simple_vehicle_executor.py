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

class SimpleVehicleExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, generator, test_validator: AbstractValidator= None, results_path: str = None):
        super().__init__(generator, test_validator, results_path)
        self._name = "SimpleVehicleExecutor"

    def _execute(self, test) -> float:

        fitness, _ = evaluate_scenario(test)

        log.info(f"Fitness: {fitness}")
        #fitness = 0 
        
        return fitness



