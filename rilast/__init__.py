from rilast.crossover.one_point_crossover import OnePointCrossover
from pymoo.operators.crossover.sbx import SBX
from rilast.mutation.obstacle_mutation import ObstacleMutation
from rilast.crossover.one_point_crossover_obstacle import OnePointCrossoverOb

from rilast.mutation.kappa_mutations import KappaMutation
from rilast.mutation.latent_mutation import LatentMutation
from pymoo.operators.mutation.pm import PM
from rilast.mutation.uniform_mutation import UniformMutation
from rilast.mutation.obstacle_mutation import LatentObstacleMutation

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES

from rilast.sampling.abstract_sampling import AbstractSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from rilast.sampling.greedy_sampling import GreedySampling

from rilast.problems.lkas_vae_problem import LKASVAEProblem
from rilast.problems.lkas_problem import LKASProblem

from rilast.executors.beam_executor import BeamExecutor
from rilast.executors.simple_vehicle_executor import SimpleVehicleExecutor
from rilast.executors.curve_executor import CurveExecutor
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from rilast.common.duplicate_removal import SimpleDuplicateElimination, LatentDuplicateElimination
from rilast.sampling.lhs_sampling import LHSSampling
ALGORITHMS = {
    "ga": GA, # Genetic Algorithm,
    "de": DE, # Differential Evolution
    "es": ES, # Evolution Strategy
    "random": RandomSearch
}

SAMPLERS = {
    "random": FloatRandomSampling,
    "lhs": LHS,
    "abstract": AbstractSampling,
    "lhs_sampling": LHSSampling,
    "greedy": GreedySampling
}

CROSSOVERS = {
    "one_point": OnePointCrossover,
    "sbx": SBX,
    "one_point_ob": OnePointCrossoverOb
}

MUTATIONS = {
    "kappa": KappaMutation,
    "latent": LatentMutation,
    "obstacle": ObstacleMutation,
    "pm": PM,
    "uniform": UniformMutation,
    "latent_obstacle": LatentObstacleMutation
}


PROBLEMS = {
    "lkas": LKASProblem,
    "lkasvae": LKASVAEProblem
}

EXECUTORS = {
    "beam": BeamExecutor,
    "simple_vehicle": SimpleVehicleExecutor,
    "curve": CurveExecutor
}

DUPLICATE_ELIMINATIONS = {
    "simple": SimpleDuplicateElimination,
    "float": LatentDuplicateElimination
}






