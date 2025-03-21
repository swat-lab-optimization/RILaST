"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-11-02
Description: script for evaluating the diversity of test scenarios
"""

import numpy as np
from rilast.generators.abstract_generator import AbstractGenerator

def calc_novelty(x_1:np.ndarray, x_2:np.ndarray, generator: AbstractGenerator) -> float:
    """
    Calculate the novelty between two states using a specified generator's comparison function.

    Args:
      x_1 (np.ndarray): the first state to compare
      x_2 (np.ndarray): the second state to compare
      generator (AbstractGenerator): the generator providing the comparison function

    Returns:
float: The novelty of the solution relative to the other solutions in the test suite.
    """

    novelty = generator.cmp_func(x_1, x_2)
    return novelty
