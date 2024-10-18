import loguru
import numpy as np
import sys

logger = loguru.logger
# logger.add("file_{time}.log")
logger.add(sys.stdout, level="TRACE")


class Point():
    count = 0
    def __init__(self, x, y, z, f, type, name="", index=None):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__f = f            # function as 4th dimension
        self.pheromone = 1.0
        self.__node_type = type       # 0 = hid , 1 = input, 2 = output
        self.__id = Point.count + 1
        self.name = name
        self.name_idx = index   # for input and output nodes
        Point.count += 1
        logger.debug(f"Creating Point({self.__id}): x={x}, y={y}, z={z}, f={f}, Type={type}, Name={name}")


    def set_pheromone(self, pheromone):
        self.pheromone = pheromone

    def get_pheromone(self):
        return self.pheromone
    
    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def get_z(self):
        return self.__z

    def set_z(self, z):
        self.__z = z
    
    def get_f(self):                              # Returns the function as a 4th dimension
        return self.__f
    
    def set_f(self, f):
        self.__f = f

    def set_id(self, id):
        self.__id = id

    def get_id(self):
        return self.__id

    def get_node_type(self):
        return self.__node_type

    def is_output(self):
        return self.__node_type == 2
    
    def coordinates(self):
        return [self.__x, self.__y, self.__z, self.__f]

    

    def distance_to(self, x, y, z, f):
        # logger.debug(f"Calculating distance from Point({self.__id}) to ({x}, {y}, {z}, {f})")
        # logger.debug(f"Point doing the distance: ({self.__x}, {self.__y}, {self.__z}, {self.__f})")
        return np.sqrt(
                        (self.__x - x) ** 2 + 
                        (self.__y - y) ** 2 + 
                        (self.__z - z) ** 2 + 
                        (self.__f - f) ** 2
                    )
    

class Space():
    def __init__(
                    self, 
                    input_names: list, 
                    output_names: list,
                    evap_rate=0.1,
                    lags=5,
    ):
        self.output_names = output_names
        self.input_names = input_names
        self.input_points = []
        self.output_points = {}
        self.points = []
        self.lag_levels = lags # Number of lag levels

        for idx, output_name in enumerate(self.output_names):
            # Output space is 1D x = [0, 1], y = 1, z = 1
            self.output_points[output_name] =  Point(
                                                x = idx / (len(input_names) - 1), 
                                                y = 1,          # y=0 and x between 0-1 and z=0
                                                z = 1,
                                                f = 0.0,
                                                type = 2,
                                                name = output_name,
                                                index = idx,
                                            )


    def add_new_points(self, new_points):
        self.points.extend(new_points)
    
    def add_input_points(self, input_points):
        self.input_points.extend(input_points)

    def evaporate_pheromone(self, evaporation_rate):
        logger.info(f"Evaporating pheromone with rate: {evaporation_rate}")
        for point in self.points:
            point.set_pheromone(point.get_pheromone() * evaporation_rate)

        for point in self.input_points: # Evaporate output points pheromone
            pheromone = point.get_pheromone() * evaporation_rate
            
        for point in self.output_points.values(): # Evaporate output points pheromone
            pheromone = point.get_pheromone() * evaporation_rate
            if pheromone < 0.1:                  # Minimum pheromone level for output points
                point.set_pheromone(0.1)
            else:
                point.set_pheromone(pheromone)

        points_to_remove = [] # Points with low pheromone
        for point in self.points: # Remove points with low pheromone (inside nodes)
            if point.get_pheromone() < 0.1:
                points_to_remove.append(point)
        for point in points_to_remove:
            self.points.remove(point)
        
        points_to_remove = [] # Points with low pheromone
        for point in self.input_points: # Remove points with low pheromone (input nodes)
            if point.get_pheromone() < 0.1:
                points_to_remove.append(point)
        for point in points_to_remove:
            self.input_points.remove(point)


    def deposited_pheromone(self,graph):
        for node in graph.nodes.values():
            node.point.pheromone+=1
            if node.point.pheromone>10:
                node.point.pheromone=10