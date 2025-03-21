from rilast.mutation.abstract_mutation import AbstractMutation
import numpy as np
import random
import copy

class UniformMutation(AbstractMutation):
    '''
    Some mutation ideas borrowed from:
    https://github.com/ERATOMMSD/frenetic-sbst21/blob/main/src/generators/random_frenet_generator.py
    '''
    def __init__(self, mut_rate: float = 0.4):
        super().__init__(mut_rate)

    def _do_mutation(self, x) -> np.ndarray:
        self.un_mut_rate = 0.2
        possible_mutations = [
            self._uniform_modification,

        ]

        mutator = np.random.choice(possible_mutations)

        return mutator(x)
    

    def _uniform_modification(self, kappas: np.ndarray) -> np.ndarray:
        # number of kappas to be modified
        #kappas = kappas.tolist()
        l_b = self.problem.xl
        u_b = self.problem.xu
        modified_kappas = copy.deepcopy(kappas)
        for i in range(len(modified_kappas)):
            if np.random.rand() < self.un_mut_rate:
                # Mutate the gene
                modified_kappas[i] = np.random.uniform(l_b[i], u_b[i])


        return modified_kappas