import abc
import typing
import numpy as np

class AbstractGenerator(abc.ABC):
    """Abstract class for all generators."""

    def __init__(self):
        """Initialize the generator.

        Args:
            config (dict): Dictionary containing the configuration parameters.
        """
        #self.size = solution_size
        self._name = "AbstractGenerator"

    @property
    @abc.abstractmethod
    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        pass

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        pass

    @property
    def name(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self._name
    '''
    @property
    def genotype(self) -> typing.List[float]:
        """Phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        pass
    '''
    @abc.abstractmethod
    def cmp_func(self, x:np.ndarray, y:np.ndarray) -> float:
        pass

    
    @abc.abstractmethod
    def genotype2phenotype(self, genotype: typing.List[float]) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get the genotype of the generator.

        Returns:
            list: Genotype of the generator.
        """
        pass
    '''
    
    def set_genotype(self, phenotype):
        """Set the phenotype of the generator.

        Args:
            phenotype (list): Phenotype of the generator.
        """
        pass
    '''
        
    @abc.abstractmethod
    def get_phenotype(self):
        """Get the genotype of the generator.

        Returns:
            list: Genotype of the generator.
        """
        pass

    @abc.abstractmethod
    def generate_random_test(self) -> (typing.List[float], bool):
        """Generate samples from the generator

        Returns:
            np.array: Generated samples.
        """
        pass

    @abc.abstractmethod
    def visualize_test(self, test: typing.List[float], save_path: str = None):
        """Visualize a test.

        Args:
            test (np.array): Test to visualize.
        """
        pass