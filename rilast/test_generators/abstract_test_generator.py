import abc
import logging
import numpy as np
import torch
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from torchvision import datasets, transforms
from rilast.train_utils import VAE
from rilast import ALGORITHMS, SAMPLERS, CROSSOVERS, MUTATIONS
from rilast.common.duplicate_removal import AbstractDuplicateElimination
from rilast.common.random_seed import get_random_seed
from rilast.train_utils.train_vae import (
    ToTensor1D,
    Normalize1D_1,
    Denormalize1D_1,
)

from rilast.train_utils.test_vae import load_model

log = logging.getLogger(__name__)


class AbstractTestGenerator(abc.ABC):
    def __init__(self, name="abstract_test_generator"):
        self._name = name

    def initialize_vae(self):
        
        self.nLat = self.config["nLat"]

        self.archive = np.load(self.config["dataset_path"])
        min = np.min(self.archive, axis=0)
        max = np.max(self.archive, axis=0)

        self.transform = Denormalize1D_1(min, max)
        self.transform_norm = transforms.Compose(
            [ToTensor1D(), Normalize1D_1(min, max)]
        )

        self.model = VAE[self.config["vae_architecture"]](self.nDim, self.nLat)

        path = self.config["model_path"]  # optimized model
        self.model = load_model(1000, self.model, path=path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def configure_algorithm(self):
        if self.crossover == "sbx":
            crossover = CROSSOVERS[self.crossover](prob=0.5, eta=3.0, vtype=float)
        else:
            crossover = CROSSOVERS[self.crossover](cross_rate=0.9)
        if self.mutation == "pm":
            mutation = MUTATIONS[self.mutation](prob=0.4, eta=3.0, vtype=float)
        else:
            mutation = MUTATIONS[self.mutation](mut_rate=0.4)

        self.method = ALGORITHMS[self.alg](
            pop_size=self.pop_size,
            n_offsprings=int(round(self.pop_size / 2)),
            sampling=SAMPLERS[self.sampl](self.generator),
            n_points_per_iteration=int(round(self.pop_size)),
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=AbstractDuplicateElimination(
                generator=self.generator, threshold=0.025
            ),
        )

    def initialize_parameters(self, alg, cross, mut):
        log.info("Starting test generation, initializing parameters")
        self.tc_stats = {}
        self.tcs = {}
        self.tcs_convergence = {}

        self.seed = get_random_seed()
        self.pop_size = self.config["pop_size"]
        log.info(f"Population size: {self.pop_size}")
        self.alg = alg
        self.sampl = "abstract"
        self.crossover = cross
        self.mutation = mut



    def run_optimization(self):
        self.res = minimize(
            self.problem,
            self.method,
            termination=get_termination(
                self.config["termination"], self.config["budget"]
            ),
            seed=self.seed,
            verbose=True,
            eliminate_duplicates=True,
            save_history=True,
        )

    @abc.abstractmethod
    def initialize_problem(self):
        pass
    @abc.abstractmethod
    def initialize_executor(self):
        pass

    def start(self, alg: str = "ga", cross: str = "sbx", mut: str = "pm"):

        self.initialize_parameters(alg, cross, mut)
        if ("vae_architecture") in self.config or ("model_path") in self.config:
            self.initialize_vae()
        self.initialize_problem()
        self.configure_algorithm()
        self.run_optimization()
        return self.res, self.executor
