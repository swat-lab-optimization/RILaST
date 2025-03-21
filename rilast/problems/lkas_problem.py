

from rilast.problems.abstract_problem import AbstractProblem
import logging #as log
from rilast.executors.abstract_executor import AbstractExecutor
import time
log = logging.getLogger(__name__)

class LKASProblem(AbstractProblem):
    def __init__(self, executor: AbstractExecutor, n_var: int=10, n_obj=1, n_ieq_constr=1, min_fitness = 0.95, xl=None, xu=None):
        super().__init__(executor, n_var, n_obj, n_ieq_constr, xl, xu)
        self.min_fitness = min_fitness
        self._name = "LKASProblem"


    def _evaluate(self, x, out, *args, **kwargs):
        test = x
        start = time.time()
        algorithm = kwargs["algorithm"]
        fitness = self.executor.execute_test(test, algorithm)
        #log.info(f"Time to evaluate: {time.time() - start}")
        #log.info(f"Fitness output: {fitness}")
        out["F"] = fitness
        out["G"] = self.min_fitness - fitness * (-1)