from pymoo.core.mutation import Mutation
import numpy as np
import abc

class AbstractMutation(Mutation, abc.ABC):
    def __init__(self, mut_rate: float = 0.4):
        super().__init__()
        self.mut_rate = mut_rate

    def _do(self, problem, X, **kwargs):

        self.problem = problem

        # for each individual
        for i in range(len(X)):

            r = np.random.random()

            # with a probabilty of 40% - change the order of characters
            if r < self.mut_rate:
                X[i] = self._do_mutation( X[i])

        return X
    

    @abc.abstractmethod
    def _do_mutation(self, x) -> np.ndarray:
        pass