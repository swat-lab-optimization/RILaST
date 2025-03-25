import abc
import logging
from aerialist.px4.obstacle import Obstacle
import os
from rilast.common.random_seed import get_random_seed
from rilast.executors.obstacle_scene_executor import ObstacleSceneExecutor
from rilast.executors.rrt_executor import RRTExecutor
from rilast.generators.latent_generator import LatentGenerator
from rilast.generators.obstacle_generator import ObstacleGenerator
from rilast.problems.obstacle_scene_problem import ObstacleSceneProblem
from rilast.test_generators.uav_test_generator import UAVTestGenerator
from rilast.validators.obstacle_scene_validator import ObstacleSceneValidator
from rilast.common.load_config import load_config


log = logging.getLogger(__name__)


class LatentUAVTestGenerator(UAVTestGenerator):
    def __init__(self, save_path="results", name="latent_uav_test_generator"):
        super().__init__(name)
        self.config = load_config("latent_uav_test_generator")

    def initialize_parameters(self, alg, cross, mut):
        log.info("Starting test generation, initializing parameters")

        self.seed = get_random_seed()
        self.time_budget = 200
        self.pop_size = self.config["Settings"]["pop_size"]
        log.info(f"Population size: {self.pop_size}, time budget: {self.time_budget}")
        self.alg = alg
        self.sampl = "abstract"
        self.crossover = cross
        self.mutation = mut

    def initialize_executor(self):

        min_size = Obstacle.Size(2, 2, 15)
        max_size = Obstacle.Size(20, 20, 25)
        min_position = Obstacle.Position(-40, 10, 0, 0)
        max_position = Obstacle.Position(30, 40, 0, 90)

        case_study = os.path.join("case_studies", "mission1.yaml")  # "case_studies/mission1.yaml"
        

        self.original_generator = ObstacleGenerator(
            min_size,
            max_size,
            min_position,
            max_position,
            case_study_file=case_study,
            max_box_num=3,
        )
        self.generator = LatentGenerator(
            self.nLat,
            0,
            1,
            self.original_generator,
            self.model,
            self.transform,
            self.transform_norm,
        )
        self.validator = ObstacleSceneValidator(
            min_size, max_size, min_position, max_position, generator=self.generator
        )

        self.executor =  ObstacleSceneExecutor(self.generator, self.validator) #FakeExecutor(self.latent_generator, validator) #
