import abc
import typing
import numpy as np
from numpy import dot
from numpy.linalg import norm
from rilast.generators.abstract_generator import AbstractGenerator
import torch.nn as nn
import torch
import time
import csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LatentGenerator(AbstractGenerator):
    """Abstract class for all generators."""

    def __init__(self, solution_size:int, mean:float = 0.0, std:float = 1.0, original_gen:AbstractGenerator = None, model: nn.Module = None, transform:object = None, transform_norm:object=None):
        """Initialize the generator.

        Args:
            config (dict): Dictionary containing the configuration parameters.
        """
        super().__init__() #solution_size
        self.size = solution_size
        self.vector = np.zeros(solution_size)
        self.mean = mean
        self.std = std
        self.orig_gen = original_gen
        self.model = model
        self.transform = transform
        self.transform_norm = transform_norm
        self.l_b = np.ones(solution_size)*-3
        self.u_b = np.ones(solution_size)*3
        self.random_test, _ = self.generate_random_test()

    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self.size
    
    @property    
    def size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self.size
    @property
    def genotype(self) -> typing.List[float]:
        """Phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        return self.vector
    
    def cmp_func(self, x:np.ndarray, y:np.ndarray) -> float:
        x = self.decode_test(x) 
        y = self.decode_test(y)
        difference = self.orig_gen.cmp_func(x, y)
        #cos_sim = dot(x, y)/(norm(x)*norm(y))
        #pos = cos_sim > 0
        #difference = 1 - abs(cos_sim)
        return difference
    def get_lb(self) -> np.ndarray:
        """Get the lower bound of the generator.

        Returns:
            np.array: Lower bound of the generator.
        """
        return self.l_b
    def get_ub(self) -> np.ndarray:
        """Get the upper bound of the generator.

        Returns:
            np.array: Upper bound of the generator.
        """
        return self.u_b
    

    def phenotype2genotype(self, phenotype):
        test  = self.orig_gen.phenotype2genotype(phenotype)
        test = self.encode_test(test)
        return test      


    def encode_test(self, test:typing.List[float]):
        input_data = self.transform_norm(test)
        input_data = input_data.unsqueeze(0)
        input_data = input_data.to(device)
        with torch.no_grad():
            test, _ = self.model.encode(input_data)
        return test.cpu().numpy()


    def decode_test(self, test: typing.List[float]) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Decode a test.

        Args:
            test (np.array): Test to decode.

        Returns:
            tuple: Decoded test.
        """
        x_eval = np.array(test, dtype=np.float32)
        x_eval = torch.from_numpy(x_eval).to(device)
        with torch.no_grad():
            test = self.model.decode(x_eval)
        test = self.transform(test.squeeze(0).cpu())
        test = test.numpy()
        return test
        
    def genotype2phenotype(self, genotype: typing.List[float]) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get the genotype of the generator.

        Returns:
            list: Genotype of the generator.
        """

        start = time.time()

        test = self.decode_test(genotype)
        result = self.orig_gen.genotype2phenotype(test)
        return result

    def set_genotype(self, phenotype):
        """Set the phenotype of the generator.

        Args:
            phenotype (list): Phenotype of the generator.
        """
        self.vector = phenotype
        
    def get_phenotype(self):
        """Get the genotype of the generator.

        Returns:
            list: Genotype of the generator.
        """
        return self.vector

    def generate_random_test(self) -> (typing.List[float], bool):
        """Generate samples from the generator

        Returns:
            np.array: Generated samples.
        """
        self.vector = np.random.normal(loc=self.mean, scale=self.std, size=self.size)
        return self.vector, True
    
    def test_dimension(self, dimension:int=0, value:float=0) -> int:
        '''
        Returns an array with a random value for the given dimension
        '''
        test = self.random_test.copy()
        random_dimension = value#np.random.normal(loc=self.mean, scale=self.std, size=1)
        test[dimension] = random_dimension
        return test
    
    def visualize_test(self, test: typing.List[float], save_path : str ="test", num:int = 0, title:str = ""):
        """Visualize a test.

        Args:
            test (np.array): Test to visualize.
        """

        if self.orig_gen is not None:




            phenotype = self.orig_gen.genotype2phenotype(test)
            #phenotype = self.orig_gen.denormilize_flattened_test(test)

            self.orig_gen.visualize_test(phenotype, save_path, num, title)
        