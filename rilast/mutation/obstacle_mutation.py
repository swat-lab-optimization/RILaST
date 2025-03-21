from rilast.mutation.abstract_mutation import AbstractMutation
import numpy as np
import random
import copy
from aerialist.px4.obstacle import Obstacle
import logging

log = logging.getLogger(__name__)
class ObstacleMutation(AbstractMutation):
    '''
    This class performs mutation on the obstacles of the test
    '''
    def __init__(self, mut_rate: float = 0.4):
        super().__init__(mut_rate)
        

    def _do_mutation(self, x) -> np.ndarray:
        problem = self.problem
        test = copy.deepcopy(x)
        possible_mutations = [
            self._random_modification, 
            self._add_obstacle,
            self._remove_obstacle
        ]
        self.gen = problem.executor.generator
        #self.lat_gen = problem.executor.generator
        self.validator = problem.executor.validator

        mutator = np.random.choice(possible_mutations)

        test_p = self.gen.genotype2phenotype(test)

        mutated_x = mutator(test_p)
        is_valid, _ = self.validator.is_valid(mutated_x)
        mutated_x = self.gen.phenotype2genotype(mutated_x)
        if is_valid:
            return mutated_x
        else:
            log.info("Mutation not valid")
            return test
        
    def _random_modification(self, test):

        obstacles_list = test.test.simulation.obstacles
        
        k = random.randint(0, len(obstacles_list)-1)
        obstacles_list.pop(k)
        found = False

        while not found:
            size = Obstacle.Size(
            l=random.choice(np.arange(self.gen.min_size.l, self.gen.max_size.l)),
            w=random.choice(np.arange(self.gen.min_size.w, self.gen.max_size.w)),
            h=random.choice(np.arange(self.gen.min_size.h, self.gen.max_size.h)),
            )
            position = Obstacle.Position(
            x=random.choice(np.arange(self.gen.min_position.x, self.gen.max_position.x)),
            y=random.choice(np.arange(self.gen.min_position.y, self.gen.max_position.y)),
            z=0,  # obstacles should always be place on the ground
            r=random.choice(np.arange(self.gen.min_position.r, self.gen.max_position.r)),
            )
            obstacle = Obstacle(size, position)

            to_include = self.gen.obstacle_fits(obstacle, obstacles_list)
            if to_include:
                obstacles_list.append(obstacle)
                found = True
        test.test.simulation.obstacles = obstacles_list
        return test


    def _add_obstacle(self, test):
        obstacles_list = test.test.simulation.obstacles
        if len(obstacles_list) < self.gen.max_box_num:

            found = False

            while not found:
                size = Obstacle.Size(
                l=random.choice(np.arange(self.gen.min_size.l, self.gen.max_size.l)),
                w=random.choice(np.arange(self.gen.min_size.w, self.gen.max_size.w)),
                h=random.choice(np.arange(self.gen.min_size.h, self.gen.max_size.h)),
                )
                position = Obstacle.Position(
                x=random.choice(np.arange(self.gen.min_position.x, self.gen.max_position.x)),
                y=random.choice(np.arange(self.gen.min_position.y, self.gen.max_position.y)),
                z=0,  # obstacles should always be place on the ground
                r=random.choice(np.arange(self.gen.min_position.r, self.gen.max_position.r)),
                )
                obstacle = Obstacle(size, position)

                to_include = self.gen.obstacle_fits(obstacle, obstacles_list)
                if to_include:
                    obstacles_list.append(obstacle)
                    found = True
        test.test.simulation.obstacles = obstacles_list
        return test

    def _remove_obstacle(self, test):
        obstacles_list = test.test.simulation.obstacles
        if len(obstacles_list) > 1:
            k = random.randint(0, len(obstacles_list)-1)
            obstacles_list.pop(k)
        test.test.simulation.obstacles = obstacles_list

        return test


class LatentObstacleMutation(ObstacleMutation):
    '''
    This class performs mutation on the obstacles of the test
    '''
    def __init__(self, mut_rate: float = 0.4):
        super().__init__(mut_rate)
        

    def _do_mutation(self, x) -> np.ndarray:
        problem = self.problem
        test = copy.deepcopy(x)
        possible_mutations = [
            self._random_modification, 
            self._add_obstacle,
            self._remove_obstacle
        ]
        self.gen = problem.executor.generator.orig_gen
        self.lat_gen = problem.executor.generator
        self.validator = problem.executor.test_validator

        mutator = np.random.choice(possible_mutations)

        test_p = self.lat_gen.genotype2phenotype(test)

        mutated_x = mutator(test_p)
        is_valid, _ = self.validator.is_valid(mutated_x)
        mutated_x = self.lat_gen.phenotype2genotype(mutated_x)
        if is_valid:
            return mutated_x
        else:
            return test
        