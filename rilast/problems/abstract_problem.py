from abc import ABC, abstractmethod
from pymoo.core.problem import ElementwiseProblem
from rilast.executors.abstract_executor import AbstractExecutor
import numpy as np
import torch.nn as nn


class AbstractVAEProblem(ElementwiseProblem, ABC):
    """
    This is the base class for performing solution evalaution
    """

    def __init__(self, executor: AbstractExecutor, vae: nn.Module, n_var: int=10, n_obj=1, n_ieq_constr=1):
        x_u = np.ones(n_var)*3
        x_l = np.ones(n_var)*(-3)
        self.executor = executor
        self.vae = vae
        self._name = "AbstractVAEProblem"

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=x_l, xu=x_u)

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):

        pass

    @property
    def name(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self._name

    # You can add non-abstract methods here if needed


class AbstractProblem(ElementwiseProblem, ABC):
    """
    This is the base class for performing solution evalaution
    """

    def __init__(self, executor: AbstractExecutor, n_var: int=10, n_obj=1, n_ieq_constr=1, xl=None, xu=None):
        self.executor = executor
        self._name = "AbstractProblem"

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):

        pass

    @property
    def name(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self._name
