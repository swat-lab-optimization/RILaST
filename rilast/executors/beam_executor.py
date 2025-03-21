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
from beamng_sim.code_pipeline.test_analysis import compute_all_features
log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130

class BeamExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, beamng_executor, generator, test_validator: AbstractValidator= None, results_path: str = None):
        super().__init__(generator, test_validator, results_path)
        self.beamng_executor = beamng_executor
        self._name = "BeamExecutor"
        self.sim_num = 0
        self.num_failures = 0
        self.min_fitness = self.beamng_executor.oob_tolerance

    def _execute(self, test) -> float:
        test_list = []#list(test)
        for i in test:
            test_list.append(list(i))
        #not_straight = min_radius(test) < MAX_RADIUS_THRESHOLD

        fitness = 0
        

        #if not_straight:

        the_test = RoadTestFactory.create_road_test(test_list)

        test_outcome, description, execution_data = self.beamng_executor.execute_test(the_test)

        log.info(f"Test outcome: {test_outcome}")

        fitness = -max([i.oob_percentage for i in execution_data])

        if "FAIL" in test_outcome:
            self.num_failures += 1


        self.test_dict[self.exec_counter]["outcome"] = test_outcome
        self.test_dict[self.exec_counter]["num_failures"] = self.num_failures
            
        

        log.info(f"Fitness: {fitness}")
        self.sim_num += 1
        self.test_dict[self.exec_counter]["sim_num"] = self.sim_num
        features = compute_all_features(the_test, execution_data)
        self.test_dict[self.exec_counter]["features"] = features
        #fitness = 0 
        
        return fitness



