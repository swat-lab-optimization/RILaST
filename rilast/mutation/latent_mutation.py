from rilast.mutation.abstract_mutation import AbstractMutation
import numpy as np
import random


class LatentMutation(AbstractMutation):
    '''
    Some mutation ideas borrowed from:
    https://github.com/ERATOMMSD/frenetic-sbst21/blob/main/src/generators/random_frenet_generator.py
    '''
    def __init__(self, mut_rate: float = 0.4):
        super().__init__(mut_rate)

    def _do_mutation(self, x) -> np.ndarray:
        
        possible_mutations = [
            self._random_modification,
            self._reverse_kappas,
            self._split_and_swap_kappas
        ]

        mutator = np.random.choice(possible_mutations)

        return mutator(x)
    

    def _random_modification(self, kappas: np.ndarray) -> np.ndarray:
        # number of kappas to be modified
        #kappas = kappas.tolist()

        k = random.randint(5, (kappas.size-2) )
        # Randomly modified kappa
        indexes = random.sample(range((kappas.size) - 1), k)
        modified_kappas = kappas[:]
        for i in indexes:
            modified_kappas[i] += random.choice(np.linspace(-0.05, 0.05))
            modified_kappas[i] = max(0.0, modified_kappas[i])
            modified_kappas[i] = min(1.0, modified_kappas[i])
        return modified_kappas
    
    def _reverse_kappas(self, kappas: np.ndarray) -> np.ndarray:
        return kappas[::-1]
    
    def _split_and_swap_kappas(self, kappas: np.ndarray) -> np.ndarray:
        return np.concatenate([kappas[int((kappas.size) / 2):], kappas[:int((kappas.size) / 2)]])