import abc
import typing
import numpy as np
import typing 
import random
from shapely.geometry import Polygon
from rilast.generators.abstract_generator import AbstractGenerator
import yaml
#import cv2
import os
import logging #as log
from rdp import rdp
import time
log = logging.getLogger(__name__)
from aerialist.px4.obstacle import Obstacle
from aerialist.px4.drone_test import DroneTest
from rilast.common.testcase import TestCase
from shapely import geometry
from numpy import dot
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import similaritymeasures
from joblib import Parallel, delayed
class ObstacleGenerator(AbstractGenerator):
    """Abstract class for all generators."""
    def __init__(self, min_size:Obstacle, max_size:Obstacle, min_position:Obstacle, max_position:Obstacle, case_study_file: str, max_box_num:int=3):
        """Initialize the generator.

        Args:
            config (dict): Dictionary containing the configuration parameters.
        """
        super().__init__()
        self.min_size = min_size #[min_size.l, min_size.w, min_size.h]
        self.max_size = max_size #[max_size.l, max_size.w, max_size.h]
        self.min_position = min_position #[min_position.x, min_position.y, 0, min_position.r]
        self.max_position = max_position #[max_position.x, max_position.y, 0, max_position.r]

        self.min_yaw = -1.57
        self.max_yaw = 1.57
        self.min_flight_height = 0
        self.max_flight_height = 15
        self.case_study = DroneTest.from_yaml(case_study_file)
        self.max_box_num = max_box_num
        self._size = self.max_box_num*6 + 1
        self._size = self.max_box_num*6 + 1
        self.l_b, self.u_b = self.get_bounds()
        self._genotype = None   
        self.novelty_name = None

    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, size):
        self._size = size
    
    @property
    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self.size#max_number_of_points

    def cmp_func(self, x, y):

        self.novelty_name = "cosine"

        cos_sim = dot(x, y) / (norm(x) * norm(y))
        difference = 1 - cos_sim
        return (difference)
    
    def cmp_out_func(self, feature1, feature2):

        feature_list = [feature1, feature2]
        feature_frames = []
        for feature in feature_list:
            x, y, z = np.array(feature[0]), np.array(feature[1]), np.array(feature[2])
            x_s = x#[::50]  
            y_s = y#[::50]  
            z_s = z#[::50] 
            x_s = self.normalize_vector(x_s, self.min_position.x, self.max_position.x)
            y_s = self.normalize_vector(y_s, self.min_position.y-15, self.max_position.y+15)
            z_s = self.normalize_vector(z_s, self.min_flight_height, self.max_flight_height)

            #feature_frame = np.column_stack((x_s, y_s, z_s))
            feature_frame = np.column_stack((x_s, y_s, z_s))
            start = time.time()
            feature_frame = rdp(feature_frame, epsilon=0.5)
            end = time.time()
            print("RDP time", end-start)
            #feature_frame = [x_s, y_s, z_s]
            feature_frames.append(feature_frame)
        
        dist = similaritymeasures.frechet_dist(
            feature_frames[0],
            feature_frames[1],
        )

        
        return dist
    
    def get_bounds(self):
        l_b = [1]
        u_b = [self.max_box_num]
        l_b_ = [self.min_size.l, self.min_size.w, self.min_size.h, self.min_position.x, self.min_position.y,  self.min_position.r]
        u_b_ = [self.max_size.l, self.max_size.w, self.max_size.h, self.max_position.x, self.max_position.y, self.max_position.r]

        for i in range(self.max_box_num):
            l_b.append(l_b_)
            u_b.append(u_b_)


        l_b = self.flatten_test_case(l_b)
        u_b = self.flatten_test_case(u_b)

        return l_b, u_b
    
    def get_lb(self):
        return self.size*[0]
    def get_ub(self):  
        return self.size*[1]
    
    
    def flatten_test_case(self, test):
        result = []
        for item in test:
            if isinstance(item, list):
                result.extend(self.flatten_test_case(item))
            else:
                result.append(item)
        return np.array(result)
    
    def get_random_box_vals(self):
        l=random.choice(np.arange(self.min_size.l, self.max_size.l))
        w=random.choice(np.arange(self.min_size.w, self.max_size.w))
        h=random.choice(np.arange(self.min_size.h, self.max_size.h))
        x=random.choice(np.arange(self.min_position.x, self.max_position.x))
        y=random.choice(np.arange(self.min_position.y, self.max_position.y))
        #z=0  # obstacles should always be place on the ground
        r=random.choice(np.arange(self.min_position.r, self.max_position.r))
        return [l, w, h, x, y, r]

    def generate_random_test(self, genotype=True):

        obstacles_list = []
        num_boxes = np.random.choice(np.arange(1, self.max_box_num+1))

        while len(obstacles_list) < (self.max_box_num):
            size = Obstacle.Size(
            l=random.choice(np.arange(self.min_size.l, self.max_size.l)),
            w=random.choice(np.arange(self.min_size.w, self.max_size.w)),
            h=random.choice(np.arange(self.min_size.h, self.max_size.h)),
            )
            position = Obstacle.Position(
            x=random.choice(np.arange(self.min_position.x, self.max_position.x)),
            y=random.choice(np.arange(self.min_position.y, self.max_position.y)),
            z=0,  # obstacles should always be place on the ground
            r=random.choice(np.arange(self.min_position.r, self.max_position.r)),
            )
            obstacle = Obstacle(size, position)

            to_include = self.obstacle_fits(obstacle, obstacles_list)
            if to_include:
                obstacles_list.append(obstacle)


        obstacles_list = obstacles_list[:num_boxes]

        #print("Genotype", self.genotype)
        the_test = TestCase(self.case_study, obstacles_list)

       # self.set_genotype(self.phenotype2genotype(the_test))
        self.genotype = self.phenotype2genotype(the_test)

        return the_test, True
    
    def normalize_vector(self, vector:np.ndarray, min_val:float, max_val:float):

        result = (vector - min_val)/(max_val - min_val)

        return result
    
    def normilize_flattened_test(self, test):
        result = (test - self.l_b)/(self.u_b - self.l_b)
        return result
    
    def denormilize_flattened_test(self, norm_test):
        result = norm_test*(self.u_b - self.l_b) + self.l_b
        return result
    
    def phenotype2genotype(self, phenotype):
        obstacles_list = phenotype.test.simulation.obstacles
        num_boxes = len(obstacles_list)
        tc = [num_boxes]
        for b in obstacles_list:
            tc.extend([b.size.l, b.size.w, b.size.h, b.position.x, b.position.y, b.position.r])

        #for r in range(num_boxes, self.max_box_num): # extent with empty values
        #    tc.extend([self.min_size.l, self.min_size.w, self.min_size.h, self.min_position.x, self.min_position.y, self.min_position.r])
        for r in range(num_boxes, self.max_box_num):
            tc.extend(self.get_random_box_vals())   

        tc = self.normilize_flattened_test(tc)

        return tc
    
    @property
    def genotype(self):
        return self._genotype
    

    @genotype.setter
    def genotype(self, genotype):
       self._genotype = genotype

    def get_genotype(self):
        return self._genotype
    
    def get_phenotype(self):
        self.phenotype = self.genotype2phenotype(self.genotype)
        return self.phenotype
    

    def resize_test(self, test):
        num_boxes = int(round(test[0]))
        #print(f"num_boxes {num_boxes}")
        test = test[1:]
        # resize test to the shape (max_box_num, 7)
        test = test.reshape(-1, 6)

        return [num_boxes, test]
        
    def genotype2phenotype(self, genotype):

        denormilized_tc = self.denormilize_flattened_test(genotype)
        #print("Denormilized tc", denormilized_tc)
        resized_tc = self.resize_test(denormilized_tc)
        num_boxes = min(resized_tc[0], self.max_box_num)
        tc = resized_tc[1]
        obstacles_list = []
        #print("Boxes num", num_boxes)
        #print("tc", tc)
        for b in range(num_boxes):
            size = Obstacle.Size(
            l=tc[b][0],
            w=tc[b][1],
            h=tc[b][2],
            )
            position = Obstacle.Position(
            x=tc[b][3],
            y=tc[b][4],
            z=0,  # obstacles should always be place on the ground
            r=tc[b][5],
            )
            obstacle = Obstacle(size, position)

            obstacles_list.append(obstacle)

        the_test = TestCase(self.case_study, obstacles_list)

        return the_test


    def obstacle_fits(self, obstacle:Obstacle, obstacles_list:list):

        new_box_geometry = obstacle.geometry#[obstacle.size.l, obstacle.size.w, obstacle.size.h]
        existing_boxes_geometry_list = [obstacle.geometry for obstacle in obstacles_list]#[obstacle.position.x, obstacle.position.y, obstacle.position.r]

        min_pos = [self.min_position.x, self.min_position.y]
        max_pos = [self.max_position.x, self.max_position.y]

        outer_polygon = geometry.Polygon([min_pos, [min_pos[0], max_pos[1]], max_pos, [max_pos[0], min_pos[1]]])
        

        for box in existing_boxes_geometry_list:
            if new_box_geometry.intersects(box):
                return False
        is_inside = new_box_geometry.within(outer_polygon)
        if not(is_inside):
            return False
        return True


    def visualize_test(self, test,  save_path:str = "test.png", num=0, title=""):
    #test.plot()
        obstacles = test.test.simulation.obstacles
        fig, ax = plt.subplots(figsize=(8,5.7))

        ax.set_xlim(self.min_position[0], self.max_position[0]+10) #80
        ax.set_ylim(self.min_position[1] -12, self.max_position[1] + 15) # 57

        area_x = (self.min_position[0] + self.max_position[1])/2
        area_y = (self.min_position[1] + self.max_position[1])/2

        area_width = self.max_position[0] - self.min_position[0]
        area_height = self.max_position[1] - self.min_position[1]

        rect = patches.Rectangle((area_x  -area_width/2, area_y - area_height/2), area_width, area_height, linewidth=1, edgecolor='black', facecolor='none', label='Obstacle area')
        ax.add_patch(rect)

        start_point = [0, 0]
        end_point = [0, 50]

        ax.scatter(start_point[0], start_point[1], c='green', label='Start point')
        ax.scatter(end_point[0], end_point[1], c='blue', label='End point')


        if obstacles is not None:
            for obst in obstacles:
                obst_patch = obst.plt_patch()
                ax.add_patch(obst_patch)
            obst_patch.set_label("obstacle")


        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16)
        plt.ioff()
        ax.set_title(title, fontsize=16)

        #ax.legend()

        if not(os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)
        final_path = os.path.join(save_path, str(num) + ".png")
        fig.savefig(final_path, bbox_inches='tight')
        log.info("Saved image to " + final_path)
        print("Saved image to " + final_path)
        plt.close(fig)

        

        

        



        

        
        

