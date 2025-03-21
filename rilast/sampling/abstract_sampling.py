from pymoo.core.sampling import Sampling
import numpy as np
from abc import ABC
from rilast.generators.abstract_generator import AbstractGenerator


class AbstractSampling(Sampling, ABC):

    def __init__(self, generator:AbstractGenerator) -> None:
        super().__init__()
        self.generator = generator


    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples), None, dtype=object)
        i = 0
        while i < n_samples:
            test, valid = self.generator.generate_random_test()
            if valid:
                test = self.generator.genotype
                X[i] = np.array(test)
                i += 1

        return X
