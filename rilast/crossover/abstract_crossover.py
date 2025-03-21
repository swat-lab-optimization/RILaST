from pymoo.core.crossover import Crossover
import numpy as np
import abc



class AbstractCrossover(Crossover, abc.ABC):
    def __init__(self, cross_rate: float = 0.9):
        

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.cross_rate = cross_rate

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            r = np.random.random()
            a, b = X[0, k], X[1, k]
            if r < self.cross_rate:
                off_a, off_b = self._do_crossover(problem, a, b)
            # get the first and the second parent
                Y[0, k], Y[1, k] = off_a, off_b
            else:
                Y[0, k], Y[1, k] = a, b  # if not crossover just copy the parents

        return Y
    
    @abc.abstractmethod
    def _do_crossover(self, problem, a, b) -> tuple: 
        pass