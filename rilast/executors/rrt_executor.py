from rilast.executors.abstract_executor import AbstractExecutor
from rilast.validators.abstract_validator import AbstractValidator
import logging
from rilast.common.rrt import RRT
#from rilast.common.rrt_star import RRTStar
log = logging.getLogger(__name__)
import time
class RRTExecutor(AbstractExecutor):
    """
    Class for evaluating a test case with an RRT planner
    
    Attributes:
        generator (Generator): The generator used to generate test scenarios.
        test_validator (AbstractValidator): The validator used to validate test scenarios.
        n_sim_evals (int): The number of simulation evaluations allowed.
    """
    def __init__(self, generator, test_validator: AbstractValidator= None):
        super().__init__(generator, test_validator)
        self.n_sim_evals = 0
        self._name = "RRTExecutor"
        self.min_fitness = 25

    def _execute(self, test) -> float:
        """
        Executes the test scenario using the RRT algorithm and returns the fitness value.
        
        Args:
            test (Test): The test scenario to be executed.
        
        Returns:
            float: The fitness value of the executed test scenario.
        """
        min_x = self.generator.min_position.x
        min_y = self.generator.min_position.y
        max_x = self.generator.max_position.x
        max_y = self.generator.max_position.y

        gx, gy = 0, max_y + 10 # define goal position

        show_animation = False#True
        fitness = 0
        tc_obstacle_list = test.test.simulation.obstacles 
        obstacle_list = []
        bonus = 0
        for obs in tc_obstacle_list:
            obstacle_list.append((obs.position.x, obs.position.y, obs.size.l, obs.size.w, obs.position.r))

        rrt = RRT(
            start=[0, 0],
            goal=[gx, gy],
            rand_area=[min_x+10, max_x-10, min_y - 15, max_y + 15],
            obstacle_list=obstacle_list,
            play_area=[min_x+10, max_x-10, min_y - 15, max_y + 15],
            robot_radius=2
        )
        path = rrt.planning(animation=show_animation)
        if path is not None: 
            fitness = -len(path)
        else:
            fitness = 0

        '''

        if fitness < -29:
            try:
                self.n_sim_evals += 1
                log.info(f"Running {self.n_sim_evals} simulation")
                test.execute()

                #if len(test.test_results) > 0:
                distances = test.get_distances()
                distance = min(distances)
                if distance < 1.5:
                    bonus += 10 # addd extra fitness if the test fails
                    num_obs = len(test.test.simulation.obstacles)
                    bonus += 1/num_obs*10 # add extra fitness if the number of obstacles is low

                    self.test_dict[self.exec_counter]["outcome"] = "FAIL"
                else:
                    self.test_dict[self.exec_counter]["outcome"] = "PASS"
                log.info(f"Minimum_distance:{(distance)}")
                self.test_dict[self.exec_counter]["metric"] = distance
                self.test_dict[self.exec_counter]["info"] = "simulation"
                test.plot()
            except Exception as e:
                self.test_dict[self.exec_counter]["info"] = "ERROR"
                log.info("Exception during test execution")
                log.info(f"{e}")

        fitness -= bonus
        '''
        return fitness
            
