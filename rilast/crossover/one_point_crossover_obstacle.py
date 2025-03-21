import numpy as np
from rilast.crossover.abstract_crossover import AbstractCrossover
import copy


class OnePointCrossoverOb(AbstractCrossover):
    def __init__(self, cross_rate: float = 0.9):
        super().__init__(cross_rate)


    def _do_crossover(self, problem, a, b) -> tuple:

        test_a = copy.deepcopy(a)
        test_b = copy.deepcopy(b)
        gen = problem.executor.generator

        try:
            validator = problem.executor.test_validator
        except:
            validator = problem.executor.validator

        test_a_p = gen.genotype2phenotype(test_a)
        test_b_p = gen.genotype2phenotype(test_b)

        obs_a = test_a_p.test.simulation.obstacles
        obs_b = test_b_p.test.simulation.obstacles


        n_ob1 = len(obs_a)
        n_ob2 = len(obs_b)
        if min(n_ob1, n_ob2) -1  > 1:
            cross_point = np.random.randint(1, min(n_ob1, n_ob2))
        else:
            cross_point = 1

        off_a_ = obs_a[:cross_point] + obs_b[cross_point:]
        off_b_ = obs_b[:cross_point] + obs_a[cross_point:]

        test_a_p.test.simulation.obstacles = off_a_
        test_b_p.test.simulation.obstacles = off_b_

        is_valid_a, _ = validator.is_valid(test_a_p)
        is_valid_b,_ = validator.is_valid(test_b_p)


        if not(is_valid_a):
            off_a = test_a
        else:
            off_a = gen.phenotype2genotype(test_a_p)
        if not(is_valid_b):
            off_b = test_b
        else:
            off_b = gen.phenotype2genotype(test_b_p)
        
        return off_a, off_b
