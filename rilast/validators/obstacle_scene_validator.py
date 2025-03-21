
from abc import ABC, abstractmethod
from rilast.validators.abstract_validator import AbstractValidator
from rilast.generators.abstract_generator import AbstractGenerator
from shapely.geometry import Polygon
import typing
import logging
import time
from aerialist.px4.obstacle import Obstacle
from shapely import geometry
log = logging.getLogger(__name__)

class ObstacleSceneValidator(AbstractValidator):
    """
    Validator for obstacle scenes in a simulation.
    """

    def __init__(self, min_size:Obstacle, max_size:Obstacle, min_position:Obstacle, max_position:Obstacle, generator: AbstractGenerator):
        super().__init__(generator)
        self.min_size = min_size #[min_size.l, min_size.w, min_size.h]
        self.max_size = max_size #[max_size.l, max_size.w, max_size.h]
        self.min_position = min_position #[min_position.x, min_position.y, 0, min_position.r]
        self.max_position = max_position 

    def is_valid(self, test) -> (bool, str):
        """
        Check if the given test is valid.

        Args:
            test: The test to be validated.

        Returns:
            A tuple containing a boolean indicating the validity of the test and a string with additional information.
        """
        #log.info(f"test {test}")
        boxes  = test.test.simulation.obstacles
        
        valid, info  = self.box_fits(boxes)

        return valid, info


    def box_fits(self, box_list):
        """
        Check if the boxes in the given list fit within the defined bounds.

        Args:
            box_list: A list of boxes to be checked.

        Returns:
            A tuple containing a boolean indicating if the boxes fit within the bounds and a string with additional information.
        """
        existing_boxes_geometry_list = [obstacle.geometry for obstacle in box_list]#[obstacle.position.x, obstacle.position.y, obstacle.position.r]

        min_pos = [self.min_position.x, self.min_position.y]
        max_pos = [self.max_position.x, self.max_position.y]

        outer_polygon = geometry.Polygon([min_pos, [min_pos[0], max_pos[1]], max_pos, [max_pos[0], min_pos[1]]])

        n = len(existing_boxes_geometry_list)

        for i in range(n):
            for j in range(i + 1, n):
                poly1 = existing_boxes_geometry_list[i]
                poly2 = existing_boxes_geometry_list[j]

                # Check if polygons intersect
                if poly1.intersects(poly2):
                    
                    return False, f"Obstacles {i} and {j} intersect."

        for inner_polygon in existing_boxes_geometry_list:
            if not(inner_polygon.within(outer_polygon)):
                return False, "Obstacle out of the defined bounds"
            

        return True, "Valid test"
