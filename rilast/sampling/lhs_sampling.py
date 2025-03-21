from pymoo.core.sampling import Sampling
import numpy as np
from abc import ABC
from rilast.generators.abstract_generator import AbstractGenerator
from rilast.validators.abstract_validator import AbstractValidator
from pymoo.operators.sampling.lhs import LHS

class LHSSampling(Sampling, ABC):

    def __init__(self, generator:AbstractGenerator, validator:AbstractValidator) -> None:
        super().__init__()
        self.generator = generator
        self.validator = validator


    def _do(self, problem, n_samples, **kwargs):
        n_samples = 10000
        X_final = np.zeros((n_samples, problem.n_var))
        m = 0
        while m < n_samples:
            X = LHS()._do(problem, n_samples, **kwargs)
            for ind in X:
                ind = list(ind)
                valid, msg = self.validator.is_valid(self.generator.genotype2phenotype(ind))
                if m >= n_samples:
                    break
                if valid:
                    X_final[m] = ind
                    m += 1

        return X
