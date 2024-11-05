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
def inverse(x):
    return np.sum([1/max(j, 0.0001) for j in x])
def negate(x):
    return np.sum(np.array(x) * -1)

def maximum(x):
    return np.max(x)
def minimum(x):
    return np.min(x)
def logic_add(x):
    return np.sum(np.array(x) > 0)
def logic_or(x):
    return np.sum(np.array(x) > 0) > 0
def exp(x):
    return np.sum(np.exp(x))
def log(x, base=np.e):
    x = np.clip(x, 1e-10, None)
    if base == np.e:
        log_values = np.sum(np.log(x))  # Natural log
    else:
        log_values = np.sum(np.log(x) / np.log(base))  # Log with custom base
    
    return log_values
def sqrt(x):
    signs = np.sign(x)
    return np.sum(signs*np.sqrt(np.abs(x)))
def square(x):
    return np.sum(np.square(np.abs(x)))
def cube(x):
    return np.sum(np.power(x, 3))
def softmax(x):
    return np.sum(np.exp(x)) / np.sum(np.exp(x))
def cosh(x):
    return np.sum(np.cosh(x))
def sinh(x):
    return np.sum(np.sinh(x))

def xor(x):
    # XOR over multiple inputs is true if an odd number of inputs are true
    return np.sum(x) % 2  # Returns 1 if the count of 1s is odd, 0 if even
def xnor(x):
    # XNOR is the complement of XOR, true if an even number of inputs are true
    return 1 - xor(x)  # Returns 1 if XOR result is 0, 0 if XOR result is 1
def ispositive(x):
    return np.sum(np.array(x) > 0)
def isnegative(x):
    return np.sum(np.array(x) < 0)
def iszero(x):
    return np.sum(np.array(x) < 0.05)


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
                    9: inverse,
                    10: negate,
                    11: maximum,
                    12: minimum,
                    13: logic_add,
                    14: logic_or,
                    15: exp,
                    16: log,
                    17: sqrt,
                    18: square,
                    19: cube,
                    20: softmax,
                    21: cosh,
                    22: sinh,
                    23: xor,
                    24: xnor,
                    25: ispositive,
                    26: isnegative,
                    27: iszero,
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
                    9: 'inverse',
                    10: 'negate',
                    11: 'maximum',
                    12: 'minimum',
                    13: 'logic_add',
                    14: 'logic_or',
                    15: 'exp',
                    16: 'log',
                    17: 'sqrt',
                    18: 'square',
                    19: 'cube',
                    20: 'softmax',
                    21: 'cosh',
                    22: 'sinh',
                    23: 'xor',
                    24: 'xnor',
                    25: 'ispositive',
                    26: 'isnegative',
                    27: 'iszero',
                }