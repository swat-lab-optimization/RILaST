"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for checking the validity of the generated road topologies
"""
import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point
from numpy.ma import arange
from shapely.geometry import Polygon
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import math
from rilast.common.vehicle_evaluate import interpolate_road
#import config as cf

rounding_precision = 3
interpolation_distance = 1
smoothness = 0
min_num_nodes = 20


def interpolate_test(the_test):
    """
    Interpolate the road points using cubic splines and ensure we handle 4F tuples for compatibility
    """
    old_x_vals = [t[0] for t in the_test]
    old_y_vals = [t[1] for t in the_test]

    # This is an approximation based on whatever input is given
    test_road_lenght = LineString([(t[0], t[1]) for t in the_test]).length
    num_nodes = int(test_road_lenght / interpolation_distance)
    if num_nodes < min_num_nodes:
        num_nodes = min_num_nodes

    assert len(old_x_vals) >= 2, "You need at leas two road points to define a road"
    assert len(old_y_vals) >= 2, "You need at leas two road points to define a road"

    if len(old_x_vals) == 2:
        # With two points the only option is a straight segment
        k = 1
    elif len(old_x_vals) == 3:
        # With three points we use an arc, using linear interpolation will result in invalid road tests
        k = 2
    else:
        # Otheriwse, use cubic splines
        k = 3

    pos_tck, pos_u = splprep([old_x_vals, old_y_vals], s=smoothness, k=k)

    step_size = 1 / num_nodes
    unew = arange(0, 1 + step_size, step_size)

    new_x_vals, new_y_vals = splev(unew, pos_tck)

    # Return the 4-tuple with default z and defatul road width
    return list(
        zip(
            [round(v, rounding_precision) for v in new_x_vals],
            [round(v, rounding_precision) for v in new_y_vals],
            [-28.0 for v in new_x_vals],
            [8.0 for v in new_x_vals],
        )
    )


def is_too_sharp(the_test, TSHD_RADIUS=47):
    """
    If the minimum radius of the test is greater than the TSHD_RADIUS, then the test is too sharp

    Args:
      the_test: the input road topology
      TSHD_RADIUS: The radius of the circle that is used to check if the test is too sharp. Defaults to
    47

    Returns:
      the boolean value of the check variable.
    """
    if TSHD_RADIUS > min_radius(the_test) > 0.0:
        check = True
        # print("Too sharp")
    else:
        check = False
    return check


def random_ends_too_close(point, remaining_points, margin=3):
    """
    Ensure that the distance between the first and last point is greater than the margin

    Args:
      points: a list of points that make up the road
      margin: the minimum distance between the first and last point. Defaults to 5

    Returns:
      A boolean value.
    """
    for p in remaining_points:
        if np.linalg.norm(np.array(point) - np.array(p)) < margin:
            return True
    return False

def ends_meet(points, margin=3):
    """
    The function checks if there are any random ends in the given list of points that are too close to
    each other within a specified margin.
    
    :param points: The `points` parameter is a list of points. Each point is represented as a tuple of
    two values, representing the x and y coordinates of the point
    :param margin: The margin parameter is used to determine how close the ends of the points can be
    before considering them too close. It is set to a default value of 3, but you can also provide a
    different value when calling the function, defaults to 3 (optional)
    :return: a boolean value. It returns True if there are two points in the given list "points" that
    are too close to each other based on the given margin value. Otherwise, it returns False.
    """
    for i in range(len(points) - 3):
        point = points[i]
        remaining_points = points[i+2:]
        if random_ends_too_close(point, remaining_points, margin):
            return True
    return False


def is_self_intersecting(points):
    """
    The function `is_self_intersecting` checks if a given set of points forms a self-intersecting road.
    
    :param points: The `points` parameter is a list of tuples representing the coordinates of points on
    a road. Each tuple should have two elements, representing the x and y coordinates of a point
    :return: a boolean value. It returns True if the given set of points forms a self-intersecting road,
    and False otherwise.
    """
    #print("Distance", np.linalg.norm( np.array(points[0]) - np.array(points[1])) )
    road = LineString([(t[0], t[1]) for t in points])
    ends_too_close = np.linalg.norm( np.array(points[0]) - np.array(points[-1])) < 2
    #if ends_too_close:
        #print("Ends too close")
        

    #or ends_meet(points, margin=1)
    if not(road.is_simple) or ends_too_close :
        #plot_test(points)
    #print("Self intersecting")
        return True
    else:
        return False




def is_valid_road(points, map_size=200, margin=5, consider_bounds=True):
    """
    If the road is not simple, or if the road is too sharp, or if the road has less than 3 points, or if
    the last point is not in range, then the road is invalid

    Args:
      points: a list of points that make up the road

    Returns:
      A boolean value.
    """

    if (len(points) < 3):
        return False, "Invalid road, less than 3 points"
    else:


        in_range = is_inside_map(points, map_size, margin)
        the_test = interpolate_test(points)
        inter_road = interpolate_road(points, int_factor=2)

        
        #plot_test(inter_road)


        if is_self_intersecting(inter_road) is True:
            return False, "Invalid road, self intersecting"
        
        if is_too_sharp(the_test) is True: #
            return False, "Invalid road, too sharp"

        if consider_bounds:
            if in_range is False:
                return False, "Invalid road, out of map bounds"

        if (len(points) < 3):
            return False, "Invalid road, less than 3 points"
        
        #turn_num = extract_turn_angles(points)
    

        return True, "Valid road"


# some of this code was taken from https://github.com/se2p/tool-competition-av
def find_circle(p1, p2, p3):
    """
    The function takes three points and returns the radius of the circle that passes through them

    Args:
      p1: the first point
      p2: the point that is the center of the circle
      p3: the point that is the furthest away from the line

    Returns:
      The radius of the circle.
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return np.inf

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    # print(radius)
    return radius


def plot_test(road_points, save_path="test.png"):
        road_points = list(road_points)

        intp_points = road_points#interpolate_road(road_points)

        fig, ax = plt.subplots(figsize=(8, 8))
        road_x = []
        road_y = []

        for p in intp_points:
            road_x.append(p[0])
            road_y.append(p[1])

        top = 200
        bottom = 0

        road_line = LineString(road_points)
        ax.plot(road_x[0], road_y[0], "ro", label="Start")
        ax.plot(road_x[1:], road_y[1:], "yo--", label="Road")
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

        ax.legend()
        #if not(os.path.exists(save_path)):
        #    os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

'''

def calculate_tangent_vector(point1, point2):
    # Calculate the tangent vector between two points
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    magnitude = np.sqrt(delta_x**2 + delta_y**2)
    
    if magnitude > 0:
        tangent_vector = (delta_x / magnitude, delta_y / magnitude)
    else:
        tangent_vector = (0, 0)  # Handle cases where two points are the same
    
    return tangent_vector

def calculate_number_of_turns(points):
    if len(points) < 3:
        return 0  # Not enough points for a curve

    tangent_vectors = [calculate_tangent_vector(points[i], points[i+1]) for i in range(len(points) - 1)]
    tangent_vectors.append(calculate_tangent_vector(points[-1], points[0]))


    number_of_turns = 0
    angle_list = []

    for i in range(1, len(tangent_vectors)):
        vector1 = tangent_vectors[i-1]
        vector2 = tangent_vectors[i]
        dot_product = np.dot(vector1, vector2)

        # Calculate the magnitudes (norms) of the vectors
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
            # Calculate the angle in radians
        angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

        # Convert the angle to degrees
        angle_degrees = np.degrees(angle_radians)
        angle_list.append(angle_degrees)


    turns_num = find_extrema(angle_list)

    second_derivative = np.gradient(np.gradient(angle_list))



    return turns_num

def find_extrema(values):
    if len(values) < 3:
        return 0  # Not enough data points to identify extrema

    extrema_count = 0

    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            extrema_count += 1  # Peak
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            extrema_count += 1  # Valley

    return extrema_count

def get_direction(first_point, second_point):
    """
    Get the direction vector from the first point to the second point.

    :param first_point: (x,y) coordinates of first point
    :param second_point: (x,y) coordinates of second point
    :return: Return the difference 2D vector (second_point-first_point)
    """
    return second_point[0] - first_point[0], second_point[1] - first_point[1]

def get_angle(first_vec, second_vec):
    """
    Returns the angle in degrees between the first and second vector.
    A left turn as positive angles whereas right turns have negatives.

    :param first_vec: First 2D vector
    :param second_vec: Second 3D vector
    :return: Angle between the vectors in degrees
    """
    a1, a2 = first_vec[0], first_vec[1]
    b1, b2 = second_vec[0], second_vec[1]

    angle_in_radians = math.atan2(b2, b1) - math.atan2(a2, a1)
    angle_in_degrees = math.degrees(angle_in_radians)

    return math.degrees(math.atan2(b2, b1)),math.degrees(math.atan2(a2, a1))#angle_in_degrees
def extract_turn_angles(road_points):
    """
    Extract angles of raod points and ad them to the instance variable.

    :param road_points: Points that define the road in the test scenario.
    :return: Angles in degrees of road's turns defined by the road points.
    """
    angles = []
    # iterate over "all" road points
    for i in range(2, len(road_points)):
        # calculate angle between previous direction and vector from
        # previous point to the current one

        point_before = road_points[i - 2]
        mid_point = road_points[i - 1]
        point_after = road_points[i]

        prev_direction = get_direction(point_before, mid_point)
        current_direction = get_direction(mid_point, point_after)

        turn_angle1, turn_angle2 = get_angle(prev_direction, current_direction)
        angles.append(turn_angle1)
        angles.append(turn_angle2)

    return angles
'''
def min_radius(x, w=5):
    """
    It takes a list of points (x) and a window size (w) and returns the minimum radius of curvature of
    the line segment defined by the points in the window

    Args:
      x: the x,y coordinates of the points
      w: window size. Defaults to 5

    Returns:
      The minimum radius of curvature of the road.
    """

    mr = np.inf
    nodes = x
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w - 1) / 2)]
        p3 = nodes[i + (w - 1)]
        radius = find_circle(p1, p2, p3)
        if radius < mr:
            mr = radius
    if mr == np.inf:
        mr = 0

    
    return mr * 3.280839895  # , mincurv


def is_inside_map(points, map_size, margin):
    """
    Take the extreme points and ensure that their distance is smaller than the map side
    """
    '''
    xs = [t[0] for t in interpolated_points]
    ys = [t[1] for t in interpolated_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return (
        0 < min_x
        or min_x > map_size
        and 0 < max_x
        or max_x > map_size
        and 0 < min_y
        or min_y > map_size
        and 0 < max_y
        or max_y > map_size
    )
    '''
    min_size = 0 + margin
    max_size = map_size - margin
    for x, y in points:
        if not (min_size <= x <= max_size and min_size <= y <= max_size):
            return False
    return True
