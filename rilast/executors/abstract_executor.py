import os
import numpy as np
import abc
import logging #as log

from abc import ABC, abstractmethod
from rilast.validators.abstract_validator import AbstractValidator
from rilast.generators.abstract_generator import AbstractGenerator
from typing import Tuple, Dict
import traceback
import time
log = logging.getLogger(__name__)
class  AbstractExecutor(ABC):
    """
    Class for evaluating the fitness of the test scenarios
    """
    def __init__(
        self,
        generator: AbstractGenerator,
        test_validator: AbstractValidator = None,
        results_path: str = None,
        algorithm = None,
    ):
        self.results_path = results_path
        self.validator = test_validator
        self.test_dict = {}
        self.generator = generator
        self._name = "AbstractExecutor"
        self.algorithm = algorithm

        if results_path:
            #logger.debug("Creating folder for storing simulation results.")
            os.makedirs(results_path, exist_ok=True)

        self.exec_counter = -1  # counts how many executions have been

    def execute_test(self, test, algorithm=None) -> Tuple[float, str]:
        """
        The function `execute_test` executes a test and returns the fitness score and information about the
        test execution.
        
        :param test: The `test` parameter in the `execute_test` method is a test case that will be executed.
        It is passed as an argument to the method
        :return: The function `execute_test` returns a tuple containing two values: `fitness` and `info`.
        """
        #logger.debug(f"Execution of a test #{self.exec_counter} (generation method: {test_dict['method']})")
        self.exec_counter += 1  # counts how many executions have been
        #self.algorithm = algorithm
        
        fitness = 0

        #if self.test_validator:
        self.test_dict[self.exec_counter] = {"test": list(test), "fitness": None, "info": None}
        genotype = test
        test = self.generator.genotype2phenotype(test)

        #log.info(f"Test: {test}")
        valid, info = self.validator.is_valid(test)
        #log.info(f"Test validity: {valid}")
        #log.info(f"Test info: {info}")
        if not valid:
            #logger.debug("The generated road is invalid")
            self.test_dict[self.exec_counter]["fitness"] = fitness
            self.test_dict[self.exec_counter]["info"] = info
            return float(fitness)

        try:
            start = time.time()
            fitness = self._execute(test)
            end = time.time()
            self.test_dict[self.exec_counter]["execution_time"] = end - start
            #log.info(f"Execution time: {end - start} seconds")
            self.test_dict[self.exec_counter]["fitness"] = fitness
            self.test_dict[self.exec_counter]["info"] = info
            self.test_dict[self.exec_counter]["timestamp"] = time.time()

        except Exception as e:
            log.info(f"Error {e} found")
            log.info(f"Error {traceback.format_exc()} found")
            #logger.error("Error during execution of test.", exc_info=True)
            self.test_dict[self.exec_counter]["info"] = "Error during execution of test"


        return float(fitness)

    @abc.abstractmethod
    def _execute(self, test) -> float:
        pass

    @property
    def name(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self._name



