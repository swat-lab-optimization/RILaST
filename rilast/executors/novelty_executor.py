import os
import numpy as np
import logging #as log
from rilast.validators.abstract_validator import AbstractValidator
from rilast.executors.abstract_executor import AbstractExecutor

log = logging.getLogger(__name__)

class NoveltyExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, generator, test_validator: AbstractValidator= None, results_path: str = None):
        super().__init__(generator, test_validator, results_path)

    def _execute(self, test, phenotype=None) -> float:
        fitness = 0

        if self.algorithm is None:
            log.error("No algorithm provided")
            return 0
        
        population = self.algorithm.pop.get("X")

        if len(population) > 0:
       

            novelty_list = []
            #test = self.generator.phenotype2genotype(test)
            #phenotype1 = self.generator.genotype2phenotype(test)
            #self.generator.visualize_test(phenotype1, save_path="novelty_test", num=0, title="Original test")

            
            for i in range(len(population)):
                
                #phenotype2 = self.generator.genotype2phenotype(population[i])
                novelty = self.generator.cmp_func(test, population[i])
                #self.generator.visualize_test(phenotype2, save_path="novelty_test", num=i+1, title=f"Test {i+1}, novelty: {novelty}")
                novelty_list.append(novelty)

            if len(novelty_list) > 0:
                fitness = -np.mean(novelty_list)

        log.info(f"Fitness: {fitness}")
        #fitness = 0 
        
        return fitness
    

