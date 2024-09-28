from sklearn.cluster import DBSCAN
import numpy as np
from search_space import Point
from node import Node
from util import get_center_of_mass
import loguru

import ipdb

logger = loguru.logger

class Graph:
    count = 1
    def __init__(   self, 
                    ants_paths,
                    eps=0.15,
                    min_samples=2,
                    lags=5,
                    space=None,
    ):
        self.id = Graph.count 
        self.space = space
        Graph.count += 1
        self.eps = eps
        self.min_samples = min_samples
        self.ants_paths = ants_paths
        self.lags = lags
        self.nodes = {}         # TODO: hide this from the user (after testing)
        self.in_nodes = []      # TODO: hide this from the user (after testing)
        self.out_nodes = []     # TODO: hide this from the user (after testing)

        # Create nodes
        self.create_nodes()

        # Clean the graph
        self.clean_graph()

        # Fix lags
        self.add_lagged_inputs()

        # Visualize the graph
        self.visualize_graph()


    def create_nodes(self, ):
        nodes_of_input_points = {}
        nodes_of_output_points = {}
        nodes = {}
        def add_node(curr_point, next_point, curr_node=None):
            if curr_node is None:
                # Check if the node already exists
                curr_node = nodes_of_input_points.get(
                                                    curr_point.get_id(), 
                                                    Node(type=curr_point.get_node_type(), point=curr_point, lags=self.lags)
                                                    )  
                nodes_of_input_points[curr_point.get_id()] = curr_node
            
            logger.debug(f"Node({curr_node.id}) - Point({curr_point.get_id()}) - Lag({curr_node.lag}) - Point_Z({curr_point.get_z()}), Node_Z({curr_node.z})")
            
            if next_point.get_node_type() == 2:
                # Check if the node already exists
                next_node = nodes_of_output_points.get(
                                                    next_point.get_id(), 
                                                    Node(type=next_point.get_node_type(), point=next_point, lags=self.lags)
                                                    )
                nodes_of_output_points[next_point.get_id()] = next_node
            elif next_point.get_node_type() == 0:
                # Check if the node already exists
                next_node = nodes.get(
                                                    next_point.get_id(), 
                                                    Node(type=next_point.get_node_type(), point=next_point, lags=self.lags)
                                    )
                nodes[next_point.get_id()] = next_node   # Add All node
            else:
                print(f"Unknown node type or Input node: {next_point.get_node_type()}")
                exit(1)
            
            # print(f"Point({curr_point.get_id()}) Node({curr_node.id}) ->", end=' ')

            

            curr_node.add_edge(next_node)                        # adding edge between nodes   
            return next_node

        # Create nodes
        for p, path in enumerate(self.ants_paths):
            logger.debug(f"Creating nodes for path: {p}")
            next_node = None
            for i in range(len(path)-1):
                next_node = add_node(
                                            curr_point=path[i], 
                                            next_point=path[i+1], 
                                            curr_node=next_node,
                                        )
            # print(f"Point({path[-1].get_id()}) Node({next_node.id})")
            # print()
            
        self.nodes = {node.id:node for node in {**nodes_of_input_points, **nodes, **nodes_of_output_points}.values()}
        self.in_nodes = list(nodes_of_input_points.values())
        self.out_nodes = list(nodes_of_output_points.values())

    def remove_node(self, node_id):
        # Remove node & all its references from the graph
        logger.debug(f"Removing node {node_id}")
        for inbound in self.nodes[node_id].inbound_edges.values():
            del self.nodes[inbound.source.id].outbound_edges[node_id]
        for outbound in self.nodes[node_id].outbound_edges.values():
            del self.nodes[outbound.target.id].inbound_edges[node_id]
        del self.nodes[node_id]

    def remove_edge(self, source_id, target_id):
        # Remove edge from the graph
        del self.nodes[source_id].outbound_edges[target_id]
        del self.nodes[target_id].inbound_edges[source_id]


    def merge_nodes(self,):
        # Merge nodes in the same cluster
        points_coordinates = np.array([node.point.coordinates() for node in self.nodes.values() if node.point.get_node_type() == 0])
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_coordinates)
        labels = db.labels_
        labels_set = set(labels)
        
        nodes = np.array([n for n in self.nodes.values() if n.point.get_node_type() == 0])  
        new_nodes = []
        del_nodes = []
        for label in labels_set:
            if label == -1:
                continue
            cluster_nodes = nodes[labels == label]
            logger.debug(f"Merging nodes in cluster {label} with {len(cluster_nodes)} nodes")
            center_of_mass_point = get_center_of_mass(
                                                        [node.point for node in cluster_nodes 
                                                            if node.point.get_node_type() == 0
                                                        ]
                                                       )
            new_cluster_point = Point(*center_of_mass_point)
            self.space.points.append(new_cluster_point)
            new_cluster_node = Node(point=new_cluster_point)
            logger.debug(f"\tNew cluster point: {new_cluster_point.get_id()} - Node: {new_cluster_node.id}")
            new_nodes.append(new_cluster_node)

            # Reassign edges to new node
            for node in cluster_nodes:
                logger.debug(f"\tReassigning edges from node {node.id} to new cluster node {new_cluster_node.id}")
                for edge in node.outbound_edges.values():
                    edge.source = new_cluster_node
                    new_cluster_node.outbound_edges[edge.target.id] = edge
                    logger.debug(f"\t\tRemoving edge between {node.id} and {edge.target.id} \
                                -- Replace with edge between {new_cluster_node.id} and {edge.target.id}")
                    del edge.target.inbound_edges[node.id]
                    edge.target.inbound_edges[new_cluster_node.id] = edge
                for edge in node.inbound_edges.values():
                    edge.target = new_cluster_node
                    new_cluster_node.inbound_edges[edge.source.id] = edge
                    logger.debug(f"\t\tRemoving edge between {edge.source.id} and {node.id} \
                                -- Replace with edge between {edge.source.id} and {new_cluster_node.id}")
                    del edge.source.outbound_edges[node.id]
                    edge.source.outbound_edges[new_cluster_node.id] = edge
                logger.debug(f"\tRemoving node {node.id}")

                del_nodes.append(node.id)
        for node_id in del_nodes:
            del self.nodes[node_id]
        for node in new_nodes:
            self.nodes[node.id] = node


    def detect_and_remove_cycles(self,):
        visited = set() # Nodes that have been visited
        stack = set()   # Nodes that are currently being visited

        def dfs(node, prev_node=None):
            visited.add(node)
            stack.add(node)
            for edge in node.outbound_edges.values():
                if edge.target not in visited:
                    if dfs(edge.target, node):
                        return True
                elif edge.target in stack:
                    # Cycle detected: Remove edge that close the cycle
                    logger.debug(f"Cycle detected: Removing Edge between {node.id} and {edge.target.id} ")
                    self.remove_edge(node.id, edge.target.id)
                    return True

            stack.remove(node)
            return False

        # Run DFS on all nodes to detect cycles
        for node in self.nodes.values():
            if node not in visited:
                dfs(node)

    def remove_dead_ends(self,):
        self.visualize_graph('before_cleaning')
        # Step 1: Remove dead ends (nodes with no outbound edges or no inbound edges except for input and output nodes)
        dead_ends = [node for node in self.nodes.values() 
                            if len(node.outbound_edges) == 0 and 
                               len(node.inbound_edges)  == 0 and
                               node.type == 0
                    ]
        for node in dead_ends:
            self.remove_node(node.id)

        # Step 2: Remove nodes not reachable from input nodes
        
        def dfs(node, from_in):
            if from_in:
                edge_list = node.outbound_edges
                reachable = reachable_from_in
            else:
                edge_list = node.inbound_edges
                reachable = reachable_from_out
            reachable.add(node)
            for edge in edge_list.values():
                if from_in:
                    if edge.target not in reachable:
                        dfs(edge.target, from_in)
                else:
                    if edge.source not in reachable:
                        dfs(edge.source, from_in)
        
        # Run DFS from input nodes
        reachable_from_in = set()
        for node in self.in_nodes:
            dfs(node, from_in=True)

        # Run DFS from output nodes
        reachable_from_out = set()
        for node in self.out_nodes:
            dfs(node, from_in=False)

        # Remove nodes not reachable from input nodes
        nodes_ids = list(self.nodes.keys())
        inreachable_from_in  = [node.id for node in self.nodes.values() if node not in reachable_from_in]
        inreachable_from_out = [node.id for node in self.nodes.values() if node not in reachable_from_out]
        inreachable = set(inreachable_from_in + inreachable_from_out)
        
        for node_id in inreachable:
            logger.debug(f"Removing node {node_id} as it is not reachable from input nodes")
            self.remove_node(node_id)


    def clean_graph(self,):
        self.merge_nodes()              # Merge nodes in proximity
        self.detect_and_remove_cycles() # Detect and remove cycles
        self.remove_dead_ends()         # Remove dead ends
        self.detect_and_remove_cycles() # Detect and remove cycles


    def add_lagged_inputs(self,):
        added_points = []
        added_nodes = []
        visited = set()

        def thrust_forward(node, prev_node):
            if node in visited:
                return
            visited.add(node)
            if node.z<prev_node.z:
                if node.z<=in_node.z:
                    new_input_lag_point = Point(
                                                    x = in_node.point.get_x(), 
                                                    y = 0, 
                                                    z = node.z, 
                                                    f = in_node.point.get_f(), 
                                                    type=1,
                                                )
                    new_input_lag_node = Node(point=new_input_lag_point, type=1)
                    new_input_lag_node.add_edge(node)
                    self.space.input_points.append(new_input_lag_point)
                    added_points.append(new_input_lag_point)
                    added_nodes.append(new_input_lag_node)
                else:
                    print(f"\tNode {node.id}({node.point.get_id()}) is above input node {in_node.id}({in_node.point.get_id()})")
                    in_node.add_edge(node)
                    

            edges = list(node.outbound_edges.values())
            for edge in edges:
                thrust_forward(edge.target, node)

        
        #iterate over input nodes to start there and reach output nodes
        for in_node in self.in_nodes:
            edges = list(in_node.outbound_edges.values())
            for edge in edges:
                thrust_forward(edge.target, in_node)

        for point in added_points:
            self.ants_paths.append([point])
        for node in added_nodes:
            self.nodes[node.id] = node
            self.in_nodes.append(node)

    
    """ 
    ************************************ Visualization *************************************** 
    """
    def visualize_graph(self, filename=None,):
        import graphviz as gv
        dot = gv.Digraph(comment='Graph Visualization')
        
        # Add nodes
        for node in self.nodes.values():
            if node.type == 0:
                dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()})")
            elif node.type==1:
                dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()})", shape='box', style='filled', fillcolor='lightblue')
            elif node.type==2:
                dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()})", shape='box', style='filled', fillcolor='lightgrey')
            else:
                print(f"Visualization: Unknown node type: Node({node.id} Type: {node.type})")
                exit(1)
        
        
        # Add edges
        for node in self.nodes.values():
            for edge in node.outbound_edges.values():
                dot.edge(str(node.id), str(edge.target.id), style='solid', color='green')

            for edge in node.inbound_edges.values():
                dot.edge(str(node.id), str(edge.source.id), style='dotted', color='red', arrowhead='none')
        
        # Render and save the graph
        if filename is None:
            filename = f"graph_{self.id}.gv"
        dot.render(filename, view=True)


    def plt_path_points(self, ax=None, size=40, save_here=False, show=False):
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True
        for i,points in enumerate(self.ants_paths):
            xs = [p.get_x() for p in points]
            ys = [p.get_y() for p in points]
            zs = [p.get_z() for p in points]
            fs = [p.get_f() for p in points]
            
            ax.scatter(xs, ys, zs, c=fs, cmap='viridis', marker='o', s=size)
            ax.plot(xs, ys, zs, color='black')#clrs[i])
            
            for i in range(len(xs) - 1):
                ax.quiver(xs[i], ys[i], zs[i], 
                xs[i+1] - xs[i], ys[i+1] - ys[i], zs[i+1] - zs[i], 
                arrow_length_ratio=0.1, color='b')
            for j in range(len(points)):
                ax.text(xs[j], ys[j], zs[j], points[j].get_id(), color='black', fontsize=10)
        if save_here:
            plt.savefig(f"graph_{self.get_id()}_paths.png")
            if show:
                plt.show()


    def plot_nodes(self, ax=None, size=40, save_here=False, show=False):
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True
        for node in self.nodes.values():
            ax.scatter(node.point.get_x(), node.point.get_y(), node.z, color='red', marker='*', s=size)
            ax.text(node.point.get_x(), node.point.get_y(), node.z, f"{node.id}({node.point.get_id()})", color='red', fontsize=10)
        if save_here:
            plt.savefig(f"graph_{self.get_id()}_nodes.png")
            if show:
                plt.show()
        