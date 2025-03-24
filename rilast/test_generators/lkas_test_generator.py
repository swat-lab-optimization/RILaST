import sys
import logging
import abc
from rilast.problems.lkas_problem import LKASProblem
from rilast.executors.beam_executor import BeamExecutor
from rilast.executors.curve_executor import CurveExecutor
from rilast.validators.road_validator import RoadValidator
from rilast.generators.kappa_generator import KappaRoadGenerator
from rilast.test_generators.abstract_test_generator import AbstractTestGenerator
from rilast.common.load_config import load_config


if sys.platform.startswith("win"):
    from beamng_sim.code_pipeline.beamng_executor import BeamngExecutor


log = logging.getLogger(__name__)


class LKASTestGenerator(AbstractTestGenerator):
    def __init__(self, save_path="results", map_size=200, name="lkas_test_generator"):
        super().__init__(name)

        self.config = load_config("lkas_test_generator")

        self.map_size = map_size
        const = 40

        self.beamng_executor = BeamngExecutor(
            save_path,
            map_size,
            oob_tolerance=0.85,
            time_budget=10800 + const,  # 8500,
            beamng_home="C:\\DIMA\\BeamNG\\BeamNG.tech.v0.26.2.0",
            beamng_user="C:\\DIMA\\BeamNG\\BeamNG.tech.v0.26.2.0_user",
            road_visualizer=None,
        )


    def initialize_executor(self):
        

        self.generator = KappaRoadGenerator(self.map_size, solution_size=self.nDim)
        self.validator = RoadValidator(self.generator, self.map_size)

        self.executor = BeamExecutor(
            self.beamng_executor, self.generator, test_validator=self.validator
        )

    def initialize_problem(self):
        self.nDim = self.config["nDim"]
        self.initialize_executor()

        self.problem = LKASProblem(
            executor=self.executor,
            n_var=self.generator.size,
            min_fitness=self.executor.min_fitness,
            xl=self.generator.get_lb(),
            xu=self.generator.get_ub(),
        )
