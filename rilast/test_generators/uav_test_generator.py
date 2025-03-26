import abc
import logging
from rilast.test_generators.abstract_test_generator import AbstractTestGenerator
from rilast.generators.obstacle_generator import ObstacleGenerator
from rilast.validators.obstacle_scene_validator import ObstacleSceneValidator
from rilast.executors.rrt_executor import RRTExecutor
from rilast.problems.obstacle_scene_problem import ObstacleSceneProblem
from rilast.executors.obstacle_scene_executor import ObstacleSceneExecutor
from rilast.common.load_config import load_config

from aerialist.px4.obstacle import Obstacle

log = logging.getLogger(__name__)


class UAVTestGenerator(AbstractTestGenerator):
    def __init__(self, save_path="results", name="uav_test_generator"):
        super().__init__(name)
        self.config = load_config("uav_test_generator")

    def initialize_executor(self):
        min_size = Obstacle.Size(2, 2, 15)
        max_size = Obstacle.Size(20, 20, 25)
        min_position = Obstacle.Position(-40, 10, 0, 0)
        max_position = Obstacle.Position(30, 40, 0, 90)

        case_study = "case_studies/mission1.yaml"
        self.generator = ObstacleGenerator(
            min_size,
            max_size,
            min_position,
            max_position,
            case_study_file=case_study,
            max_box_num=3,
        )
        self.validator = ObstacleSceneValidator(
            min_size, max_size, min_position, max_position, generator=self.generator
        )

        self.executor = ObstacleSceneExecutor(self.generator, self.validator)

    def initialize_problem(self):
        self.nDim = self.config["nDim"]
        self.initialize_executor()

        self.problem = ObstacleSceneProblem(
            self.executor,
            n_var=self.generator.size,
            l_b=self.generator.l_b,
            u_p=self.generator.u_b,
            min_fitness=self.executor.min_fitness,
        )
