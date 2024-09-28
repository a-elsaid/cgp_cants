import loguru
import numpy as np

logger = loguru.logger
# logger.add("file_{time}.log")


class Point():
    count = 0
    def __init__(self, x, y, z, f=0, type=0):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__f = f            # function as 4th dimension
        self.__pheromone = 1.0
        self.__node_type = type       # 0 = hid , 1 = input, 2 = output
        self.__id = Point.count + 1
        Point.count += 1
        logger.debug(f"Creating Point({self.__id}): x={x}, y={y}, z={z}, f={f}, type={type}")


    def set_pheromone(self, pheromone):
        self.pheromone = pheromone

    def get_pheromone(self):
        return self.__pheromone
    
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
        logger.debug(f"Calculating distance from Point({self.__id}) to ({x}, {y}, {z}, {f})")
        logger.debug(f"Point doing the distance: ({self.__x}, {self.__y}, {self.__z}, {self.__f})")
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
                    functions: dict, 
                    evap_rate=0.1,
                    lags=5,
    ):
        self.input_names = input_names
        self.output_names = output_names
        self.functions = functions
        self.input_points = {}
        self.output_points = {}
        self.points = []
        self.lag_levels = lags # Number of lag levels
        self.evaporation_rate = evap_rate

        for idx, input_name in enumerate(self.input_names):
            self.input_points[input_name] = Point(
                                                    x = idx / (len(input_names) - 1), 
                                                    y = 0.0,              # y=0 and x between 0-1 and z=0.5
                                                    z = 0.5,
                                                    f = 0.0,
                                                    type = 1,
                                                )    

        for idx, output_name in enumerate(self.output_names):
            self.output_points[output_name] = Point(
                                                        x = idx / (len(input_names) - 1), 
                                                        y = 1,          # y=0 and x between 0-1 and z=0
                                                        z = 1,
                                                        f = 0.0,
                                                        type = 2,
                                                    )    

        def evaporate(self, rate):
            for point in self.points:
                point.set_pheromone(point.get_pheromone() * self.evaporation_rate)
                if point.get_pheromone()< 0.01:
                    self.points.remove(point)
            
            for point in self.input_points.values():
                pheromone = point.get_pheromone() * self.evaporation_rate
                if pheromone < 0.1:
                    point.set_pheromone(0.1)
                else:
                    point.set_pheromone(pheromone)
                
            for point in self.output_points.values():
                pheromone = point.get_pheromone() * self.evaporation_rate
                if pheromone < 0.1:
                    point.set_pheromone(0.1)
                else:
                    point.set_pheromone(pheromone)
                
            for point in self.output_points.values():
                pheromone = point.get_pheromone() * self.evaporation_rate
                if pheromone < 0.1:
                    point.set_pheromone(0.1)
                else:
                    point.set_pheromone(pheromone)
            

