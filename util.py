import numpy as np

def get_center_of_mass(points):
        center_pheromone_x = 0.0
        center_pheromone_y = 0.0
        center_pheromone_z = 0.0
        center_pheromone_f = 0.0
        total_pheromone = 0.0
        for point in points:
            pheromone = point.get_pheromone()
            center_pheromone_x += point.get_x() * pheromone
            center_pheromone_y += point.get_y() * pheromone
            center_pheromone_z += point.get_z() * pheromone
            center_pheromone_f += point.get_f() * pheromone
            total_pheromone += pheromone
        if total_pheromone == 0:
            return None
        return [
            center_pheromone_x / total_pheromone,
            center_pheromone_y / total_pheromone,
            center_pheromone_z / total_pheromone,
            center_pheromone_f / total_pheromone,
        ]


def dbscan(points, epsilon, min_pts):
    def distance(p1, p2):
        d = np.sqrt(
                    np.power(
                        np.sum(
                                np.subtract(p1, p2),
                                2
                        )
                    )
            )

        return d
        
    def expand_cluster(cluster, point, neighbors, points, epsilon, min_pts, visited):
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                new_neighbors = get_neighbors(neighbor, points, epsilon)
                
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)
            
            if neighbor not in [p for p in cluster]:
                cluster.append(neighbor)


    def get_neighbors(point, points, epsilon):
        neighbors = []
        for p in points:
            if distance(point, p) <= epsilon:
                neighbors.append(p)
        return neighbors

    clusters = []
    visited = set()
    
    for point in points:
        if point in visited:
            continue
        visited.add(point)
        
        neighbors = get_neighbors(point, points, epsilon)
        
        if len(neighbors) < min_pts:
            continue
        
        cluster = [point]
        expand_cluster(cluster, point, neighbors, points, epsilon, min_pts, visited)
        clusters.append(cluster)
    
    return clusters

# Functions
def sin(x):
    return np.sum(list(map(np.sin, x)))
def cos(x):
    return np.sum(list(map(np.cos, x)))
def tan(x):
    return np.sum(list(map(np.tan, x)))
def sigmoid(x):
    def sig_fun(x):
        x = np.float64(x)
        return 1 / (1 + np.exp(-x))
    return np.sum(list(map(sig_fun, x)))
def relu(x):
    def relu_fun(x):
        return max(0, x)
    return relu_fun(np.sum(x))
def leaky_relu(x):
    def leaky_relu_fun(x):
        return max(0.01*x, x)
    return leaky_relu_fun(np.sum(x))
def add(x):
    return np.sum(x)
def multiply(x):
    return np.prod(x)
def tanh(x):
    return np.tanh(np.sum(x))




function_dict = {
                    0: add, 
                    1: multiply, 
                    2: sin, 
                    3: cos,
                    4: tan,
                    5: sigmoid,
                    6: relu,
                    7: leaky_relu,
                    8: tanh,
                }
function_names = {
                    0: 'add', 
                    1: 'multiply', 
                    2: 'sin', 
                    3: 'cos',
                    4: 'tan',
                    5: 'sigmoid',
                    6: 'relu',
                    7: 'leaky_relu',
                    8: 'tanh',
                }
