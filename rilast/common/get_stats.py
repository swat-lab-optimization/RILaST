"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for getting the stats of the optimization algorithm
"""
from itertools import combinations
import logging #as log
from rilast.common.calc_novelty import calc_novelty
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
log = logging.getLogger(__name__)
def get_stats(res):
    """
    It takes the results of the optimization and returns a dictionary with the fitness, novelty, and
    convergence of the optimization

    Args:
      res: the result of the optimization
      problem: the problem we're trying to solve

    Returns:
      A dictionary with the fitness, novelty, and convergence of the results.
    """
    algorithm = res.algorithm
    generator = algorithm.problem.executor.generator

    if isinstance(algorithm, RandomSearch):
        n_offsprings = algorithm.n_points_per_iteration
        pop_size = n_offsprings
    else:
        pop_size = algorithm.pop_size

    #pop_size = algorithm.pop_size
    res_dict = {}
    gen = len(res.history) - 1
    population_fitness = res.history[gen].pop.get("F")*(-1)
    population_fitness = [float(i) for i in population_fitness]
    #if algo != "nsga2" and algo != "rigaa":
    #    population = sorted(population, key=lambda x: x[0], reverse=True)
    #for i in range(pop_size):

        # result = res.history[gen].pop.get("F")[i][0]
    #    results.append(population[i][0])

    #gen = len(res.history) - 1
    novelty_list = []
    population = res.history[gen].pop.get("X")
    #if algo != "nsga2" and algo != "rigaa":
    #    test_population = sorted(
    #        test_population, key=lambda x: abs(x[0].fitness), reverse=True
    #    )
    for i in combinations(range(0, pop_size), 2):
        current1 = population[i[0]]  # res.history[gen].pop.get("X")[i[0]]
        current2 = population[i[1]]  # res.history[gen].pop.get("X")[i[1]]
        nov = calc_novelty(current1, current2, generator)
        novelty_list.append(nov)
    novelty = sum(novelty_list) / len(novelty_list)

    log.info("The highest fitness found: %f", max(population_fitness))
    log.info("Average diversity: %f", novelty)
    res_dict["fitness"] = population_fitness
    res_dict["novelty"] = novelty
    res_dict["exec_time"] = float(res.exec_time)

    return res_dict


