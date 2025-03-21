from abc import ABC, abstractmethod
import torch
import logging #as log
from pymoo.core.problem import ElementwiseProblem
from rilast.problems.abstract_problem import AbstractVAEProblem
from rilast.executors.abstract_executor import AbstractExecutor
import numpy as np
import torch.nn as nn
import time
log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LKASVAEProblem(AbstractVAEProblem):
    def __init__(self, executor: AbstractExecutor, transform:object, vae: nn.Module, n_var: int=10, n_obj=1, n_ieq_constr=1, min_fitness = 0.95):
        super().__init__(executor, vae, n_var, n_obj, n_ieq_constr)
        self.transform = transform
        self.min_fitness = min_fitness
        self._name = "LKASVAEProblem"


    def _evaluate(self, x, out, *args, **kwargs):
        '''
        start = time.time()
        x_eval = np.array(x, dtype=np.float32)
        x_eval = torch.from_numpy(x_eval).to(device)
        with torch.no_grad():
            test = self.vae.decode(x_eval)
        test = self.transform(test.cpu())
        test = test.numpy()
        end = time.time() - start
        log.info(f"Test generation time: {end} seconds")
        '''
        fitness = self.executor.execute_test(x)
        log.info(f"Fitness output: {fitness}")
        out["F"] = fitness
        out["G"] = self.min_fitness - fitness * (-1)






