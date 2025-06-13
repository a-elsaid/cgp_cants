import matplotlib.pyplot as plt
import numpy as np
from search_space import Point
from graph import Graph
from colony import Colony
from ant import Ant
from search_space import Space
from util import function_dict, function_names
import sys
import loguru

logger = loguru.logger
logger.remove()

import ipdb
clrs = [
        'aliceblue',
        # 'antiquewhite',
        'aqua',
        'aquamarine',
        'azure',
        'beige',
        'bisque',
        'blanchedalmond',
        'blue',
        'blueviolet',
        'brown',
        'burlywood',
        'cadetblue',
        'chartreuse',
        'chocolate',
        'coral',
        'cornflowerblue',
        'cornsilk',
        'crimson',
    ]

def create_random_points(start, end, num_points=10):
    points = [
                Point(
                        type=0, 
                        x=np.random.random(),
                        y=np.random.random(),
                        z=np.random.random(),
                        f=np.random.random(),
                )
                for _ in range(num_points)
                ]
    points = np.array([start]+sorted(points, key=lambda x: x.get_y())+[end])
    return points


def get_random_paths():
    path1 = create_random_points(
                                    Point(type=1, x=0.50, y=0.00, z=0.00, f=0.30), 
                                    Point(type=2, x=0.10, y=1.00, z=0.00, f=0.44),
                                )

    path2 = create_random_points(
                                    Point(type=1, x=0.90, y=0.00, z=0.00, f=0.20), 
                                    Point(type=2, x=0.70, y=1.00, z=0.00, f=0.44),
                                )

    path3 = create_random_points(
                                    Point(type=1, x=0.50, y=0.00, z=0.00, f=0.20), 
                                    Point(type=2, x=0.70, y=1.00, z=0.00, f=0.99),
                                )

    return [path1, path2, path3]

def test_graph(paths):
    # Create random points
    path1 = create_random_points(
                                    Point(type=1, x=0.50, y=0.00, z=0.00, f=0.30), 
                                    Point(type=2, x=0.10, y=1.00, z=0.00, f=0.44),
                                )

    path2 = create_random_points(
                                    Point(type=1, x=0.90, y=0.00, z=0.00, f=0.20), 
                                    Point(type=2, x=0.70, y=1.00, z=0.00, f=0.44),
                                )

    path3 = create_random_points(
                                    Point(type=1, x=0.50, y=0.00, z=0.00, f=0.20), 
                                    Point(type=2, x=0.70, y=1.00, z=0.00, f=0.99),
                                )

    ants_paths = [path1, path2, path3]

    input_names = ['input1', 'input2', 'input3', 'input4']
    output_names = ['output1', 'output2']
    space = Space(
                    input_names=input_names, 
                    output_names=output_names,
    )

    # Create another graph object
    graph = Graph(ants_paths, space=space)

    # Plot the paths
    plot_points(ants_paths, graph_id=graph.get_id())


    print("Graph Passed Loops Test:",  graph.check_validity())

    plot_point([node.point for node in graph.get_nodes()], clr='black', size=100, graph_id=graph.get_id())
    visualize_graph(graph, "graph")

    # Test the graph object
    assert len(graph.get_in_nodes()) == 2
    assert len(graph.get_out_nodes()) == 2

    
    print("Number of inside nodes: ", len(graph.get_inside_nodes()), " Total nodes: ", graph.total_nodes)
    assert len(graph.get_inside_nodes()) == graph.total_nodes - 4 # 2 input + 2 output


    print("Graph testing passed!")

    

def test_ant():
    input_names = ['input1', 'input2', 'input3', 'input4']
    output_names = ['output1', 'output2']
    space = Space(
                    input_names=input_names, 
                    output_names=output_names,
                    functions=function_dict,
    )
    ant = Ant(space)
    ant.march()
    paths = [ant.path]  
    graph = Graph(paths)
    plot_points(paths, graph_id=graph.get_id())
    visualize_graph(graph)

def test_ants(num_ants=10):
    input_names = ['input1', 'input2', 'input3', 'input4']
    output_names = ['output1', 'output2']
    space = Space(
                    input_names=input_names, 
                    output_names=output_names,
    )
    ants = [Ant(space) for _ in range(num_ants)]
    for ant in ants:
        ant.march()
    paths = [ant.path for ant in ants]  
    graph = Graph(ants_paths = paths, space=space)

    fig = plt.figure(figsize=(10, 10))  
    ax = fig.add_subplot(111, projection='3d')
    graph.plot_path_points(ax)
    graph.plot_nodes(ax)
    graph.plot_pheromones(ax)
    plt.savefig(f"graph_{graph.id}.png")
    graph.visualize_graph("graph")

    in_data = np.random.rand(100, 4)
    out_data = np.random.rand(100, 2)
    graph.feed_forward(in_data)

def test_colony():
    input_names = ['input1', 'input2', 'input3', 'input4']
    output_names = ['output1', 'output2']
    data = np.random.rand(100, 4)
    target = np.random.rand(100, 2)
    colony = Colony(
                    num_ants=10, 
                    population_size=10, 
                    input_names=input_names,
                    output_names=output_names,
                    train_input=data,
                    train_target=target,
                    num_itrs=1000,
    )
    colony.life()


def test_ant():
    input_names = ['input1', 'input2', 'input3', 'input4']
    output_names = ['output1', 'output2']
    space = Space(
                    input_names=input_names, 
                    output_names=output_names,
    )
    ant = Ant(space)
    best_fit = None
    for _ in range(60):
        ant.march()
        fit = np.random.uniform(0, 1)
        ant.update_best_behaviors(fit)
        ant.evolve_behavior() 
        change = "BETTER"
        if best_fit is None or best_fit > fit:
            change = "WORSE"
            best_fit = fit
        print(f"{_:3d}) Fitness: {fit:.5f} -- {change:15s} Explore Rate: {ant.explore_rate:.5f} - Sense Range: {ant.sense_range:.5f}")


def check_backpropagation():
    # Create another graph object
    input_names = ['input1', 'input2', 'input3', 'input4']
    output_names = ['output1', 'output2']
    space = Space(
                    input_names=input_names, 
                    output_names=output_names,
    )
    ants = [Ant(space) for _ in range(2)]
    for ant in ants:
        ant.march()
    paths = [ant.path for ant in ants]  
    graph = Graph(ants_paths = paths, space=space)
    graph.visualize_graph("graph")

    in_data = np.random.rand(7, 4)
    out_data = np.random.rand(7, 2)
    err = graph.evaluate(in_data, out_data)

    edges = graph.get_edges()
    grads = np.array([edges.grad for edges in edges])/(len(in_data)-graph.lags)
    
    # Function to compute the numerical gradient using finite differences
    def numerical_gradient(f, edges, epsilon=1e-4):
        num_grad = np.zeros(len(edges))
        print("Checking gradients") 
        for i in range(len(edges)):
            orig_edge_weight = edges[i].weight
            
            edges[i].weight = orig_edge_weight + epsilon
            # print("Edge weight: ", edges[i].weight, " Orig edge weight: ", orig_edge_weight)
            _, loss1 = graph.evaluate(in_data, out_data)
            loss1 = np.mean(loss1**2)
            
            edges[i].weight = orig_edge_weight - epsilon
            # print("Edge weight: ", edges[i].weight, " Orig edge weight: ", orig_edge_weight)
            _, loss2 = graph.evaluate(in_data, out_data)
            loss2 = np.mean(loss2**2)

            # print("Loss1: ", loss1, " Loss2: ", loss2)
            num_grad[i] = (loss1 - loss2) / (2 * epsilon)
            edges[i].weight = orig_edge_weight


            # exit()
        return num_grad
    
    num_grads = numerical_gradient(graph.mse, edges)
    print("Grads: ", grads)
    print("Num grads: ", num_grads)

    print(f"Difference: {np.average(grads-num_grads)}")

    # Compare the numerical gradient with the backprop gradient
    diff = np.linalg.norm(num_grads - grads) / np.linalg.norm(num_grads + grads)
    
    # Check if the difference is small enough
    if diff < 1e-7:
        print("Gradient check passed!")
    else:
        print(f"Gradient check failed. Difference: {diff}")



if __name__ == "__main__":
    logger.add(sys.stdout, level="INFO")
    # test_graph() # Uncomment to run the graph test
    # test_ant()  # Uncomment to run the ant test
    # test_ant()  # Uncomment to run the ant test
    # test_ants(10)  # Uncomment to run the ants test
    # test_colony()  # Uncomment to run the colony test
    check_backpropagation()  # Uncomment to run the backpropagation test
