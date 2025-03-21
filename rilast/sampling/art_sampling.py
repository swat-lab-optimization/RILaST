from pymoo.core.sampling import Sampling
import numpy as np
from abc import ABC
from rilast.generators.abstract_generator import AbstractGenerator
from rilast.sampling.abstract_sampling import AbstractSampling
from rilast.executors.abstract_executor import AbstractExecutor


class GreedySampling(AbstractSampling):

    def __init__(self, generator:AbstractGenerator, greedy_executor:AbstractExecutor) -> None:
        super().__init__(generator)
        #self.generator = generator
        self.executor = greedy_executor


    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples), None, dtype=object)
        i = 0
        while i < n_samples:
            test = self._select_best_of_k(10)
            X[i] = test
            i += 1

        return X
    
    def _select_best_of_k(self, k:int):
        best_fitness = 0
        best_phenotype = []

        for i in range(k):
            test, valid = self.generator.generate_random_test()
            phenotype = self.generator.genotype

            fitness = self.executor.execute_test(test)
            if fitness < best_fitness:
                best_fitness = fitness
                best_phenotype = phenotype

        return best_phenotype
