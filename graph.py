from sklearn.cluster import DBSCAN
import numpy as np
from search_space import Point
from node import Node
from util import get_center_of_mass
import loguru
import graphviz as gv
from time import time
import sys
import torch

import ipdb

logger = loguru.logger
logger.add(sys.stdout, level="INFO")

class Graph:
    count = 1
    def __init__(   self, 
                    ants_paths,
                    eps=0.25,
                    min_samples=2,
                    lags=5,
                    space=None,
                    colony_id=None,
                    use_torch=True,
    ):
        self.id = Graph.count 
        self.__use_torch = use_torch
        self.space = space
        self.colony_id = colony_id
        Graph.count += 1
        self.eps = eps
        self.min_samples = min_samples
        self.ants_paths = ants_paths
        self.lags = lags
        self.nodes = {}         # TODO: hide this from the user (after testing)
        self.in_nodes = []      # TODO: hide this from the user (after testing)
        self.out_nodes = []     # TODO: hide this from the user (after testing)
        self.added_points = []
        self.added_in_points = []

        # Create nodes
        self.create_nodes()

        # Merge nodes in the same cluster
        self.merge_nodes()

        # Clean the graph
        self.clean_graph()

        # Fix lags
        self.add_lagged_inputs()

        # ipdb.set_trace()

    
    def get_edges(self,):
        return [edge for node in self.nodes.values() for edge in node.outbound_edges.values()]

    def create_nodes(self, ):
        nodes_of_input_points = {}
        nodes_of_output_points = {}
        nodes = {}
        def add_node(curr_point, next_point, curr_node=None):
            if curr_node is None:
                # Check if the node already exists
                curr_node = nodes_of_input_points.get(
                                                    curr_point.name, 
                                                    Node(type=curr_point.get_node_type(), point=curr_point, lags=self.lags)
                                                    )  
                nodes_of_input_points[curr_point.name] = curr_node
            
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
            logger.debug(f"Creating nodes for Path: {p}")
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
        logger.debug(f"Removing Node({node_id})")
        for inbound in self.nodes[node_id].inbound_edges.values():
            if inbound.source.id in self.nodes:
                del self.nodes[inbound.source.id].outbound_edges[node_id]
        for outbound in self.nodes[node_id].outbound_edges.values():
            if outbound.target.id in self.nodes:
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
            logger.debug(f"Merging nodes in Cluster {label} with {len(cluster_nodes)} nodes")
            center_of_mass_point = get_center_of_mass(
                                                        [node.point for node in cluster_nodes 
                                                            if node.point.get_node_type() == 0
                                                        ]
                                                       )
            new_cluster_point = Point(*center_of_mass_point, type=0)
            # self.space.points.append(new_cluster_point)   # Adding new points in the colony to decouple the graph from the colony
            self.added_points.append(new_cluster_point)
            new_cluster_node = Node(point=new_cluster_point, lags=self.lags)
            logger.debug(f"\tNew cluster Point({new_cluster_point.get_id()}) - Node({new_cluster_node.id})")
            new_nodes.append(new_cluster_node)

            # Reassign edges to new node
            logger.debug(f"Reassigning functions and edges for Cluster({label})")
            for node in cluster_nodes:
                logger.debug(f"\tReassigning functions from Node({node.id}) \
                        to new cluster Node({new_cluster_node.id})")
                new_cluster_node.add_functions(node.functions)
                logger.debug(f"\tReassigning edges from Node({node.id}) \
                        to new cluster Node({new_cluster_node.id})")
                for edge in node.outbound_edges.values():
                    edge.source = new_cluster_node
                    new_cluster_node.outbound_edges[edge.target.id] = edge
                    logger.debug(f"\t\tRemoving edge between Node({node.id}) and Node({edge.target.id}) \
                                -- Replace with edge between Node({new_cluster_node.id}) and Node({edge.target.id})")
                    del edge.target.inbound_edges[node.id]
                    edge.target.inbound_edges[new_cluster_node.id] = edge
                for edge in node.inbound_edges.values():
                    edge.target = new_cluster_node
                    new_cluster_node.inbound_edges[edge.source.id] = edge
                    logger.debug(f"\t\tRemoving edge between Node({edge.source.id}) and Node({node.id}) \
                                -- Replace with edge between Node({edge.source.id}) and Node({new_cluster_node.id})")
                    del edge.source.outbound_edges[node.id]
                    edge.source.outbound_edges[new_cluster_node.id] = edge
                logger.debug(f"\tRemoving Node({node.id})")

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
                    logger.debug(f"Cycle detected: Removing Edge between Node({node.id}) and Node({edge.target.id})")
                    self.remove_edge(node.id, edge.target.id)
                    return True

            stack.remove(node)
            return False

        # Run DFS on all nodes to detect cycles
        for node in self.nodes.values():
            if node not in visited:
                cycle_detected = dfs(node)
                while cycle_detected:
                    visited = set()
                    stack = set()
                    cycle_detected = dfs(node)

    def remove_dead_ends(self,):
        # Step 1: Remove dead ends (nodes with no outbound edges or no inbound edges except for input and output nodes)
        dead_ends = [node for node in self.nodes.values() 
                            if (
                                len(node.outbound_edges) == 0 or 
                                len(node.inbound_edges)  == 0
                               ) and
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
        unreachable_from_in  = [node.id for node in self.nodes.values() if node not in reachable_from_in]
        unreachable_from_out = [node.id for node in self.nodes.values() if node not in reachable_from_out]
        unreachable = set(unreachable_from_in + unreachable_from_out)
        logger.trace(f"Unreachable nodes: {[node_id for node_id in unreachable]}")
        
        for node_id in unreachable:
            logger.debug(f"Removing Node({node_id}) as it is not reachable from input nodes")
            self.remove_node(node_id)

        # Step 3: Remove unconnected input and output nodes
        node_to_remove = []
        for node in self.in_nodes:
            if node.id not in self.nodes:
                node_to_remove.append(node)
        for node in node_to_remove:
            self.in_nodes.remove(node)

        node_to_remove = []             
        for node in self.out_nodes:
            if node.id not in self.nodes:
                node_to_remove.append(node)
        for node in node_to_remove:
                self.out_nodes.remove(node)

    def clean_graph(self,):
        self.detect_and_remove_cycles() # Detect and remove cycles
        self.remove_dead_ends()         # Remove dead ends


    def add_lagged_inputs(self,):
        added_points = []
        added_nodes = []
        visited = set()

        def thrust_forward(node, prev_node, in_node):
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
                    logger.trace(f"LAG POINT: Adding point: Node({new_input_lag_point.get_id()})")
                    input_index = round(new_input_lag_point.get_x()*(len(self.space.input_names)-1))
                    new_input_lag_point.name = self.space.input_names[input_index]
                    new_input_lag_point.name_idx = input_index
                    logger.trace(f"LAG NODE: Adding node: Node({new_input_lag_point.get_id()})")
                    new_input_lag_node = Node(point=new_input_lag_point, type=1, lags=self.lags)
                    new_input_lag_node.add_edge(node)
                    # self.space.input_points.append(new_input_lag_point)   #adding new points in colony to decouple the graph from the colony
                    self.added_in_points.append(new_input_lag_point)
                    # added_points.append(new_input_lag_point)              #adding new points in colony to decouple the graph from the colony
                    self.added_points.append(new_input_lag_point)
                    added_nodes.append(new_input_lag_node)
                else:
                    logger.trace(f"LAG EDGE: Adding edge between Node({in_node.id}) and Node({node.id}), Type: {in_node.type}")
                    in_node.add_edge(node)

                    

            edges = list(node.outbound_edges.values())
            for edge in edges:
                thrust_forward(edge.target, node, in_node)

        
        #iterate over input nodes to start there and reach output nodes
        for in_node in self.in_nodes:
            edges = list(in_node.outbound_edges.values())
            for edge in edges:
                thrust_forward(edge.target, in_node, in_node)

        # for point in added_points:
        #     self.space.points.append(point) #adding new points in colony to decouple the graph from the colony
        for node in added_nodes:
            self.nodes[node.id] = node
            self.in_nodes.append(node)
    

    def cross_entropy(self, y_true, y_pred, prt=False):
        # y_true = np.array(y_true)
        # y_pred = np.array(y_pred)
        if prt: print(f"Prediction: {y_pred.item():.7f}, GT: {y_true.item():.7f}")
        ce = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        d_ce = - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        logger.trace(f"Cross Entropy: {ce} -- Derivative: {d_ce}")
        return ce, d_ce


    def mse(self, y_true, y_pred, prt=False):
        # mse = np.array(y_true) - np.array(y_pred)
        err = y_true - y_pred
        if prt: print(f"Prediction: {y_pred.item():.7f}, GT: {y_true.item():.7f}, Err: {err.item():.7f}")
        mse = (err*err)/2
        d_mse = err
        logger.trace(f"Mean Squared Error: {mse} -- Derivative: {d_mse}")
        return mse, d_mse


    def softmax_output(self, x, prt=False):
        # x = np.array(x)
        if prt: print(f"Softmax Input: {x}")
        exp_x = torch.exp(x - torch.max(x))
        softmax_x = exp_x / torch.sum(exp_x)
        d_softmax_x = softmax_x * (1 - softmax_x)
        logger.trace(f"Softmax Output: {softmax_x} -- Derivative: {d_softmax_x}")
        return softmax_x, d_softmax_x


    def evaluate(self, data, cal_gradient=True):
        # target = target[:-self.lags]                         # Remove the first lags values
        num_epochs = 15
        input = torch.tensor(data.train_input, requires_grad=True, dtype=torch.float32)
        target = torch.tensor(data.train_output, requires_grad=True, dtype=torch.float32) # TODO: change dtype
        
        def thrust(input, target, prt=False):
            preds, errors, d_errors = [], [], []
            input_data, target_data = input, target
            self.feed_forward(input_data)
            out = self.get_output()
            out, d_softmax_x = self.softmax_output(out, prt=prt)

            preds.append(out)
            err, d_err = self.cross_entropy(target_data, out, prt)
            # if self.__use_torch:
                # err.retain_grad()
            errors.append(err)
            d_errors.append(d_err)
            for node, e in zip(self.out_nodes, d_err):
                node.d_err = e
            if cal_gradient:
                self.feed_backward(err)


            #for i in range(0, len(input) - self.lags):
            #    input_data, target_data = input[i:i+max(1,self.lags)], target[i+self.lags]
            #    self.feed_forward(input_data)
            #    out = self.get_output()
            #    out, d_softmax_x = self.softmax_output(out, prt=prt)

            #    preds.append(out)
            #    err, d_err = self.cross_entropy(target_data, out, prt)
                # if self.__use_torch:
                    # err.retain_grad()
            #    errors.append(err)
            #    d_errors.append(d_err)
            #    for node, e in zip(self.out_nodes, d_err):
            #        node.d_err = e
            #    if cal_gradient:
            #        self.feed_backward(err)
            # self.feed_backward(torch.stack(errors))
            return preds, errors, d_errors
            
        for epoch in range(1,num_epochs+1):
            preds, errors, d_errors = thrust(input, target, prt=False)
                    
            train_errors = torch.stack(errors) if self.__use_torch else np.array(errors)
            train_d_errors = torch.stack(d_errors) if self.__use_torch else np.array(d_errors)
            train_preds = torch.stack(preds) if self.__use_torch else np.array(preds)
            if num_epochs>1: 
                logger.info(f"\tColony({self.colony_id:3d}):: Epoch: {(epoch):3d}/{num_epochs} Epoch MSE: {torch.mean(train_errors):.6e}")
                if torch.mean(train_errors) > 10:
                    break

        input = torch.tensor(data.test_input, requires_grad=True, dtype=torch.float32)
        target = torch.tensor(data.test_output, requires_grad=True, dtype=torch.float32)
        preds, errors, d_errors = thrust(input, target, prt=False)

        test_errors   = torch.stack(errors) if self.__use_torch else np.array(errors)
        test_d_errors = torch.stack(d_errors) if self.__use_torch else np.array(d_errors)
        test_preds    = torch.stack(preds) if self.__use_torch else np.array(preds)
        logger.info(f"Colony({self.colony_id}):: Training Log Loss: {torch.mean(train_errors):.6e}, TEST Log Loss: {torch.mean(test_errors):.6e}")

        return torch.mean(test_errors).detach().numpy(), d_errors

    def feed_forward(self, in_data):
        for i, node in enumerate(self.in_nodes):
            node.recieve_fire(in_data[node.lag][node.point.name_idx])

    def feed_backward(self,err):
        if self.__use_torch:
            torch.mean(err).backward()
        for node in self.out_nodes:
            node.update_weights(lr=0.01)

    def get_output(self):
        return torch.stack([node.node_value for node in self.out_nodes])









    
    """ 
    ************************************ Visualization *************************************** 
    """
    def visualize_graph(self, filename=None,):
        dot = gv.Digraph(comment='Graph Visualization')
        
        # Add nodes
        for node in self.nodes.values():
            if node.type == 0:
                dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()})")
            elif node.type==1:
                dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()}) {node.point.name}", shape='box', style='filled', fillcolor='lightblue')
            elif node.type==2:
                dot.node(str(node.id), label=f"N({node.id}) - p({node.point.get_id()}) {node.point.name}", shape='box', style='filled', fillcolor='lightgrey')
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
        dot.render(filename, view=False)


    def plot_path_points(self, ax=None, size=40, save_here=False, show=False, plt=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True
            clear_here = True
        else:
            clear_here = False
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
            plt.savefig(f"graph_{self.id}_paths.png")
            if show:
                plt.show()
        if clear_here:
            plt.cla()
            plt.clf()
            plt.close('all')


    def plot_nodes(self, ax=None, size=40, save_here=False, show=False, plt=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True
        for node in self.nodes.values():
            ax.scatter(node.point.get_x(), node.point.get_y(), node.z, color='red', marker='*', s=size)
            ax.text(node.point.get_x(), node.point.get_y(), node.z, f"{node.id}({node.point.get_id()})", color='red', fontsize=10)
        if save_here:
            plt.savefig(f"graph_{self.id}_nodes.png")
            if show:
                plt.show()
        
    def plot_pheromones(self, ax=None, size=40, save_here=False, show=False, plt=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')
            save_here = True
        for point in self.space.points:
            sc = ax.scatter(
                                point.get_x(), 
                                point.get_y(), 
                                point.get_z(), 
                                s=point.get_f()*100, 
                                c=point.get_pheromone(), 
                                cmap='viridis', 
                                alpha=0.2, 
                                marker='o'
                            )
