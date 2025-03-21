from pymoo.core.duplicate import ElementwiseDuplicateElimination
import numpy as np

import math
import logging 
from numpy import dot
from numpy.linalg import norm
from rilast.generators.abstract_generator import AbstractGenerator
log = logging.getLogger(__name__)
import time



class AbstractDuplicateElimination(ElementwiseDuplicateElimination):


    def __init__(self, generator:AbstractGenerator, threshold:float = 0.15):
        super().__init__()
        self.generator = generator
        self.threshold = threshold

    def is_equal(self, a, b):
        vector1 = np.array(a.X)
        vector2 = np.array(b.X)
        #euclidean_distance = np.linalg.norm(vector1 - vector2)
        #rand_num = int(time.time() * 1000000)
        difference = self.generator.cmp_func(vector1, vector2)
        #if difference < self.threshold and difference > 0.1:
        #    log.info(f"Duplicate detected: {difference}")
        #    self.generator.visualize_test(vector1, "duplicates", num = "1_" + str(rand_num), title = str(difference) )
        #    self.generator.visualize_test(vector2, "duplicates", num = "2_" + str(rand_num), title = str(difference) )
        return difference < self.threshold



class SimpleDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        vector1 = np.array(a.X)
        vector2 = np.array(b.X)
        #euclidean_distance = np.linalg.norm(vector1 - vector2)

        cos_sim = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
        return cos_sim > 0.85
        #return np.array_equal(vector1, vector2)



class LatentDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        vector1 = np.array(a.X)
        vector2 = np.array(b.X)
        #euclidean_distance = np.linalg.norm(vector1 - vector2)
        cos_sim = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
        return cos_sim > 0.95
        #return np.array_equal(vector1, vector2)
