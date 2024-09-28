from search_space import Space, Point
import numpy as np
from util import get_center_of_mass
import loguru
logger = loguru.logger

class Ant():
    def __init__(self, space: Space):
        self.x = 0
        self.y = 0
        self.z = 0
        self.f = 0
        self.sense_radius = np.random.uniform(0.1, 0.50)
        self.mortality = np.random.uniform(0.1, 0.5)
        self.expore_rate = np.random.uniform(0.15, 1)
        self.space = space
        self.path = []

    def update_position(self, point):
        self.x = point.get_x()
        self.y = point.get_y()
        self.z = point.get_z()
        self.f = point.get_f()
    
    def create_new_point(self, ):
        logger.debug(f"Creating new point")
        
        def create_coordinates(coord, lower_bound=0):
            return np.random.uniform(
                                        max(0, coord - lower_bound), 
                                        min(coord + self.sense_radius, 1)
                                    )
        # Create a new point based on the current position and pheromone levels
        lower_bound = self.sense_radius
        x = create_coordinates(self.x, lower_bound=lower_bound)
        z = create_coordinates(self.z, lower_bound=lower_bound)
        f = create_coordinates(self.f, lower_bound=lower_bound)
        if z<=self.z:
            lower_bound = 0
        y = create_coordinates(self.y, lower_bound=0)

        if y>=0.98:
            return self.pick_output()

        new_point =  Point( x, y, z, f)
        self.update_position(new_point)
        return new_point

    def pick_point(self, points):
        # Check if any of the output points are within the sense radius
        
        
        nearby_output_points = [
                    o_point 
                            for o_point in self.space.output_points.values() 
                            if o_point.distance_to(
                                                    self.x, 
                                                    self.y, 
                                                    self.z, 
                                                    self.f,
                                                ) <= self.sense_radius
                            ]
        


        if np.random.uniform() < self.expore_rate:
            # Explore: pick a random point within the sense radius
            return self.create_new_point()   
        else:
            # Exploit: if there are nearby output points, pick one of them
            if nearby_output_points and np.random.uniform() < self.expore_rate:
                return self.pick_output()

            # Exploit: choose the point with the highest pheromone level
            pheromone_points = []
            for point in points:
                # new points should not be in the lower left quadrant
                # wrt to the current position
                # becuase this will generate a loop
                if not(point.get_y() < self.y and point.get_z() < self.z):
                    if point.distance_to(self.x, self.y, self.z, self.f) <= self.sense_radius:
                        pheromone_points.append(point)

            # Calculate the center of mass of pheromones
            pheromone_center = get_center_of_mass(points)
            if pheromone_center is None:
                return self.create_new_point()
            else:
                point = Point(*pheromone_center)
                self.update_position(point)
                return point
            
    def in_out_pick_point(self, type):
        if type == 1:
            points = self.space.input_points.values()
        elif type == 2:
            points = self.space.output_points.values()
        if np.random.uniform() < self.expore_rate:
            # Explore: pick a random point within the sense radius
            point = np.random.choice(list(points), size=1)[0]
            self.update_position(point)
            return point
        else:
            # Exploit: choose the point with the highest pheromone level
            expectation = np.random.uniform(0, max([p.get_pheromone() for p in points]))  # TODO: Adjust this value to be the max pheromone level
            for point in points:
                if point.get_pheromone() > expectation:
                    self.update_position(point)
                    return point
                expectation -= point.get_pheromone()
            
         
    def pick_input(self,):
        point = self.in_out_pick_point(type=1)
        logger.debug(f"Picking input: {point.get_id()}")
        return point

    def pick_output(self,):
        logger.debug("Picking output") 
        return self.in_out_pick_point(type=2)

    def choose_point(self,):
        points_within_radius = []
        for point in self.space.points:
            if (
                    not(point.get_x() < self.y and point.get_z() > self.z) and 
                    point.distance_to(self.x, self.y, self.z, self.f) <= self.sense_radius
                ):
                points_within_radius.append(point)
        chosen_point = self.pick_point(points_within_radius)
        self.x, self.y, self.z, self.f = chosen_point.coordinates()
        return chosen_point


    def fix_path_z_values(self,):
        logger.debug("Fixing path z values (no later points should have a lower z value)")
        lowest_z = 1
        for i in range(len(self.path)-1, -1, -1):
            point = self.path[i]
            if point.get_z() <= lowest_z:
                lowest_z = point.get_z()
            else:
                new_point = Point(point.get_x(), point.get_y(), lowest_z, point.get_f(), type=self.path[i].get_node_type())
                self.path[i] = new_point

    def march(self,):
        self.path.append(self.pick_input())
        while not self.path[-1].is_output():
            self.path.append(self.choose_point())
        # self.fix_path_z_values()
