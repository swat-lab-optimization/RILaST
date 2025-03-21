import os
import numpy as np
import logging #as log
from rilast.validators.abstract_validator import AbstractValidator
from rilast.executors.abstract_executor import AbstractExecutor
import time
log = logging.getLogger(__name__)
class NoveltySearchExecutor(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, generator, test_validator: AbstractValidator= None, results_path: str = None, novelty_threshold=0.15, max_archive_size=200):
        super().__init__(generator, test_validator, results_path)

        self.archive = []
        self.k_val = 10#len(self.algorithm.pop.get("X"))
        self.treshold = novelty_threshold
        self.n_evals = 0
        self.individuals_added = 0
        self.total_evals = 0
        self.total_added = 0
        self.max_archive_size = max_archive_size

        

    def _execute(self, test, phenotype=None) -> float:


        fitness = 0

        if self.algorithm is None:
            log.error("No algorithm provided")
            return 0
        
        population = list(self.algorithm.pop.get("X"))
        self.n_offspring = self.algorithm.n_offsprings


        novelty_list = []



        if len(population) > 0:
            #phenotype1 = self.generator.genotype2phenotype(test)
            #phenotype1 = self.generator.denormilize_flattened_test(test)

            #if self.total_evals > 400:
            #self.generator.visualize_test(phenotype1, save_path="novelty_search_cos_uav", num=0, title=f"Original test")
            #population_phenotypes = [self.generator.genotype2phenotype(ind) for ind in population]
            total_population = population + self.archive
            #self_novelty = self.generator.cmp_func(test, test)
            #log.info(f"Self novelty: {self_novelty}")

            for i in range(len(total_population)):
                
                #phenotype2 = self.generator.genotype2phenotype(total_population[i])
                #phenotype2 = self.generator.denormilize_flattened_test(total_population[i])
                #start = time.time()
                novelty = self.generator.cmp_func(test, total_population[i])
                #print(f"Time novelty: {time.time() - start}")
                #if self.total_evals > 400:
                #self.generator.visualize_test(phenotype2, save_path="novelty_search_cos_uav", num=i+1, title=f"Test {i+1}, novelty: {novelty}")
                novelty_list.append(novelty)

        if len(novelty_list) > 0:

            novelty_list.sort(key=lambda x: x)

            novelty = np.mean(novelty_list[1:self.k_val])
            #print(f"Novelty: {novelty}")
            fitness = -novelty
            
            if novelty > self.treshold:
                self.archive.append(test)
                self.individuals_added += 1
                self.total_added += 1

            self.n_evals += 1
            self.total_evals += 1
            if self.n_evals >= self.n_offspring:
                self.n_evals = 0
                
                if self.individuals_added > 30:
                    self.treshold *= 1.05
                    log.info(f"Threshold increased to {self.treshold}")

                self.individuals_added = 0

            if self.total_evals >= 500:
                if self.total_added < 4:
                    self.treshold *= 0.95
                self.total_evals = 0

        #log.info(f"Fitness: {fitness}")

        #fitness = 0 
        
        return fitness

class NoveltySearchExecutorPhen(AbstractExecutor):
    """
    Class for executing the test scenarios in the BeamNG simulator
    """
    def __init__(self, generator, test_validator: AbstractValidator= None, results_path: str = None, novelty_threshold=0.15):
        super().__init__(generator, test_validator, results_path)

        self.archive = []
        self.k_val = 30#15
        self.treshold = novelty_threshold
        self.n_evals = 0
        self.individuals_added = 0
        self.total_evals = 0
        self.total_added = 0

    def _execute(self, test, phenotype=None) -> float:


        fitness = 0

        if self.algorithm is None:
            log.error("No algorithm provided")
            return 0
        
        population = list(self.algorithm.pop.get("X"))
        self.n_offspring = self.algorithm.n_offsprings


        novelty_list = []
        phenotype1 = self.generator.genotype2phenotype(test)

        #self.generator.visualize_test(phenotype1, save_path="novelty_test_haus", num=0, title=f"Original test")


        if len(population) > 0:
            #population_phenotypes = [self.generator.genotype2phenotype(ind) for ind in population]
            total_population = population + self.archive

            for i in range(len(total_population)):
                
                phenotype2 = self.generator.genotype2phenotype(total_population[i])
                novelty = self.generator.cmp_func3(phenotype1, phenotype2)
                #self.generator.visualize_test(phenotype2, save_path="novelty_test_haus", num=i+1, title=f"Test {i+1}, novelty: {novelty}")
                novelty_list.append(novelty)

        if len(novelty_list) > 0:

            novelty_list.sort(key=lambda x: x)

            novelty = np.mean(novelty_list[1:self.k_val])
            #print(f"Novelty: {novelty}")
            fitness = -novelty
            
            if novelty > self.treshold:
                self.archive.append(test)
                self.individuals_added += 1
                self.total_added += 1

            self.n_evals += 1
            self.total_evals += 1
            if self.n_evals >= self.n_offspring:
                self.n_evals = 0
                
                if self.individuals_added > 30:
                    self.treshold *= 1.05
                    log.info(f"Threshold increased to {self.treshold}")

                self.individuals_added = 0

            if self.total_evals >= 500:
                if self.total_added < 4:
                    self.treshold *= 0.95
                self.total_evals = 0

        #log.info(f"Fitness: {fitness}")

        #fitness = 0 
        
        return fitness
    

