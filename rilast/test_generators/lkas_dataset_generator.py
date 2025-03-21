import sys
import logging
from rilast.executors.curve_executor import CurveExecutor
from rilast.validators.road_validator import RoadValidator
from rilast.generators.kappa_generator import KappaRoadGenerator
from rilast.test_generators.lkas_test_generator import LKASTestGenerator
from rilast.common.load_config import load_config

log = logging.getLogger(__name__)


class LKASDatasetGenerator(LKASTestGenerator):
    def __init__(self, save_path="results", map_size=200, name="lkas_dataset_generator"):
        super().__init__(save_path, map_size, name)

        self.config = load_config("lkas_dataset_generator")


    def initialize_executor(self):
        

        self.generator = KappaRoadGenerator(self.map_size, solution_size=self.nDim)
        self.validator = RoadValidator(self.generator, self.map_size)

        self.executor = CurveExecutor(self.generator, test_validator=self.validator)
