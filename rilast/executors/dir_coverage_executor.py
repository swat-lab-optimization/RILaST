import os
import numpy as np
import logging #as log
from rilast.validators.abstract_validator import AbstractValidator
from rilast.executors.abstract_executor import AbstractExecutor
from beamng_sim.code_pipeline.tests_generation import RoadTestFactory
from rilast.common.road_validity_check import min_radius
from beamng_sim.code_pipeline.test_analysis import direction_coverage, direction_coverage_klk
log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130
MIN_RADIUS_THRESHOLD = 47

class DirCovExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, generator, test_validator: AbstractValidator= None, results_path: str = None):
        super().__init__(generator, test_validator, results_path)

    def _execute(self, test) -> float:

        _, dir_cov = direction_coverage_klk(list(test))

        fitness = -dir_cov

        log.info(f"Fitness: {fitness}")
        #fitness = 0 
        
        return fitness
    

