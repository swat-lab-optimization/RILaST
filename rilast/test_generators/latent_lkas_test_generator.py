# Standard library imports
import logging

# Third-party imports


# Local project imports


from rilast.common.load_config import load_config
from rilast.executors.beam_executor import BeamExecutor
from rilast.executors.curve_executor import CurveExecutor
from rilast.generators.kappa_generator import KappaRoadGenerator
from rilast.generators.latent_generator import LatentGenerator
from rilast.problems.lkas_vae_problem import LKASVAEProblem
from rilast.test_generators.lkas_test_generator import LKASTestGenerator

from rilast.validators.road_validator import RoadValidator

# BeamNG-specific imports
from beamng_sim.code_pipeline.beamng_executor import BeamngExecutor


log = logging.getLogger(__name__)


class LatentLKASTestGenerator(LKASTestGenerator):
    def __init__(self, save_path="results", map_size=200):
        super().__init__(
            save_path=save_path, map_size=map_size, name="latent_lkas_test_generator"
        )

        self.config = load_config("latent_lkas_test_generator")

    def initialize_executor(self):

        
        self.original_generator = KappaRoadGenerator(self.map_size, solution_size=self.nDim)
        self.generator = LatentGenerator(
            self.nLat, 0, 1, self.original_generator, self.model, self.transform, self.transform_norm
        )
        self.validator = RoadValidator(self.generator, self.map_size)
        self.executor = BeamExecutor(
         self.beamng_executor, self.generator, test_validator=self.validator)

        #self.executor = CurveExecutor(self.generator, test_validator=self.validator)

