from descartes import PolygonPatch
import numpy as np
import matplotlib.pyplot as plt 
import typing
import os
from shapely.geometry import LineString
from sklearn.metrics.pairwise import cosine_similarity

from rilast.generators.abstract_generator import AbstractGenerator

from rilast.common.road_validity_check import is_valid_road

from rilast.common.road_validity_check import min_radius

from rilast.common.road_validity_check import is_inside_map

from rilast.common.car_road import Map
from rilast.common.road_validity_check import is_valid_road
from rilast.common.vehicle_evaluate import interpolate_road
class AmbieGenRoadGenerator(AbstractGenerator):
    '''
    Class to generate a road based on a kappa function.
    Part of the code is based on the following repository:
    '''
    def __init__(self, map_size:int, solution_size: int):
        super().__init__()

        self.solution_size = solution_size
        self.min_len = 5
        self.max_len = 30
        self.min_angle = 10
        self.max_angle = 90
        self.map_size = map_size
        self.road_poiunts = []
        self.scenario = []
    
    @property
    def size(self):
        return self.solution_size


    @property
    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self.size#max_number_of_points

    @property
    def genotype(self) -> typing.List[float]:
        """Phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        return self.scenario
    
    def set_genotype(self, genotype: typing.List[float]):
        """Phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        self.scenario = genotype
    
    def genotype2phenotype(self, genotype: typing.List[float]) -> typing.Tuple[np.ndarray, np.ndarray]:
        self.set_genotype(genotype)
        phenotype = self.get_phenotype()
        return phenotype
    def cmp_func(self, x, y):
        difference = 1 - abs(cosine_similarity([x], [y])[0][0])
        return difference
    

    def cmp_out_func(self, feature1, feature2):
        distance = np.linalg.norm(np.array(feature1) - np.array(feature2))
        return distance

    
    def get_phenotype(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get the genotype of the generator.

        Returns:
            list: Genotype of the generator.
        """
        map = Map(self.map_size)
        road_points, scenario = map.get_points_from_states(self.genotype)

        return road_points

    def generate_random_test(self) -> (typing.List[float], bool):

        actions = list(range(0, 3))
        lengths = list(range(self.min_len, self.max_len))
        angles = list(range(self.min_angle, self.max_angle))

        map_size = self.map_size

        #while abs(int(fitness)) == 0:  # ensures that the generated road is valid
        done = False
        test_map = Map(map_size)
        while not done:
            action = np.random.choice(actions)
            if action == 0:
                length = np.random.choice(lengths)
                done = not (test_map.go_straight(length))
            elif action == 1:
                angle = np.random.choice(angles)
                done = not (test_map.turn_right(angle))
            elif action == 2:
                angle = np.random.choice(angles)
                done = not (test_map.turn_left(angle))
        scenario = test_map.scenario

        map = Map(map_size)
        road_points, scenario = map.get_points_from_states(scenario)
        road_points = interpolate_road(road_points)

        self.road_poiunts = road_points
        self.scenario = scenario

        valid = is_valid_road(road_points, map_size)

        return road_points, valid
        
    def visualize_test(self, road_points : np.ndarray, save_path : str ="test", num:int = 0, title:str = ""):
        """
        It takes a list of states, and plots the road and the car path

        Args:
          states: a list of tuples, each tuple is a state of the car.
          save_path: The path to save the image to. Defaults to test.png
        """
        road_points = list(road_points)

        intp_points = road_points#interpolate_road(road_points)

        fig, ax = plt.subplots(figsize=(8, 8))
        road_x = []
        road_y = []

        for p in intp_points:
            road_x.append(p[0])
            road_y.append(p[1])

        top = self.map_size 
        bottom = 0

        road_line = LineString(road_points)
        ax.plot(road_x, road_y, "yo--", label="Road")
        # Plot the road as a line with custom styling
        #ax.plot(*road_line.xy, color='gray', linewidth=10.0, solid_capstyle='round', zorder=4)       
        road_poly = LineString([(t[0], t[1]) for t in intp_points]).buffer(
            4.0, cap_style=2, join_style=2
        )
        road_patch = PolygonPatch(
            (road_poly), fc="gray", ec="dimgray"
        )  # ec='#555555', alpha=0.5, zorder=4)
        ax.add_patch(road_patch)
        
        # Set axis limits to show the entire road
        ax.set_xlim(road_line.bounds[0], road_line.bounds[2])
        ax.set_ylim(road_line.bounds[1], road_line.bounds[3])

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16)
        ax.set_ylim(bottom, top)
        plt.ioff()
        ax.set_xlim(bottom, top)
        ax.set_title(title, fontsize=16)

        ax.legend()
        if not(os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + "\\" + str(num) + ".png", bbox_inches='tight')
        plt.close(fig)