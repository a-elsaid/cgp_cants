import matplotlib.pyplot as plt
import numpy as np
from search_space import Point
from graph import Graph
from ant import Ant
from search_space import Space
import ipdb

function_dict = {0: 'add', 1: 'subtract', 2: 'multiply', 3: 'divide'}
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

def create_random_points(start, end, num_ponds=10):
    points = [
                Point(
                        type=0, 
                        x=np.random.random(),
                        y=np.random.random(),
                        z=np.random.random(),
                        f=np.random.random(),
                )
                for _ in range(num_ponds)
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

    # Create another graph object
    graph = Graph(ants_paths)

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
                    functions=function_dict,
    )
    ants = [Ant(space) for _ in range(num_ants)]
    for ant in ants:
        ant.march()
    paths = [ant.path for ant in ants]  
    graph = Graph(ants_paths = paths, space=space)

    fig = plt.figure(figsize=(10, 10))  
    ax = fig.add_subplot(111, projection='3d')
    graph.plt_path_points(ax)
    graph.plot_nodes(ax)
    plt.savefig(f"graph_{graph.id}.png")
    plt.show()

if __name__ == "__main__":
    # test_graph() # Uncomment to run the graph test
    # test_ant()  # Uncomment to run the ant test
    test_ants(50)  # Uncomment to run the ants test
