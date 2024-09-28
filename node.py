import numpy as np
from search_space import Point
from math import ceil, floor
import loguru

logger = loguru.logger

function_dict = {0: 'add', 1: 'subtract', 2: 'multiply', 3: 'divide'}

class Edge():
    id = 0
    def __init__(self, source, target):
        self.id = Edge.id + 1
        self.source = source
        self.target = target
        self.weight = np.random.uniform(-1, 1)


class Node():
    count = 0
    def __init__(self, type=0, point: Point = None, lags=5):
        self.id = Node.count + 1
        Node.count += 1
        self.lag = None
        self.__cluster = None
        self.type = type                   # 0 = normal, 1 = input, 2 = output
        self.functions = {}
        self.backfire = 0.0
        self.forefire = 0.0
        self.recieved_fire = 0
        self.recieved_backfire = 0
        self.point = point
        self.inbound_edges = {}
        self.outbound_edges = {}
        self.active = False
        self.pick_node_functions(function_dict)
        self.adjust_lag(lags=lags)    # Adjust lag levels based on the z value of the point

    def get_cluseter(self):
        return self.__cluster

    def set_cluster(self, cluster):
        self.__cluster = cluster

    def add_functions(self, functions):
        self.functions.update(functions)

    def pick_functions(self, function_dict):
        functions_coof = self.point.get_f() * 10
        func1_key = max(1, floor(functions_coof))
        func2_key = min(ceil(functions_coof), max(function_dict.keys()))
        self.funtions.append(function_dict[func1_key])
        if func1_key != func2_key:
            self.funtions.append(function_dict[func2_key])
        
    def compare_corr(self, node):
        return  (
                    self.point.get_x() == node.point.get_x() and 
                    self.point.get_y() == node.point.get_y() and 
                    self.point.get_z() == node.point.get_z()
        )

    
    def pick_node_functions(self, function_dict):
        func_coord = self.point.get_f()
        func_id = (len(function_dict)-1) * func_coord
        if int(func_id) == func_id:
            func_id = int(func_id)
            self.functions.update({func_id: function_dict[func_id]})
        else:
            prev_func_id = int(func_id - 1)
            next_func_id = int(func_id + 1)
            self.functions.update(
                                    {
                                        prev_func_id: function_dict[prev_func_id], 
                                        next_func_id: function_dict[next_func_id],
                                    }
            )


    def add_edge(self, to_node):
        edge = Edge(self, to_node)
        if to_node.id == self.id:
            return
        self.outbound_edges[to_node.id] = edge
        to_node.inbound_edges[self.id]  = edge


    def fire (self, ):
        for edge in self.outbound_edges:
            edge.target.recieved_fire(self.forefire * edge.weight)
        self.recieved_fire = 0
        self.forefire = 0.0

    def backfire (self, ):
        for edge in self.inbound_edges:
            edge.source.recieve_backfire(self.backfire * edge.weight)
            edge.weight += 0.1 * self.backfire * edge.source.forefire
        self.recieved_backfire = 0
        self.backfire = 0.0
        

    def recieved_fire (self, value):
        self.recivced_fire+= 1
        self.forefire+= value
        if self.recivce_fire == len(self.inbound_edges):
            self.fire()

    def recieve_backfire (self, value):
        self.recivced_backfire+= 1
        self.recieved_backfire+= value
        if self.recivce_backfire == len(self.outbound_edges):
            self.recieved_backfire = 0
            self.backfire()

    def fix_edges_loops(self,) -> None:
        edges_to_remove = []
        # for out_edge in self.outbound_edges:
        #     for in_node_edge in out_edge.target.outbound_edges:
        #         if out_edge.source == in_node_edge.target and out_edge.target == in_node_edge.source:
        #             self.point.set_z((in_node_edge.source.point.get_z() + 0.5)%1)

        for o_edge in self.outbound_edges:
            for out_node_o_edge in o_edge.target.outbound_edges:
                if out_node_o_edge.target == self:
                    edges_to_remove.append(o_edge)

        for i, edge in enumerate(self.inbound_edges): 
            if edge.source == edge.target:          #remove self loops
                if edge not in edges_to_remove:
                    edges_to_remove.append(edge)
            for o_edge in self.inbound_edges[i+1:]: # remove redundant edges
                if edge.source == o_edge.source and edge.target == o_edge.target:
                    if o_edge not in edges_to_remove:
                        edges_to_remove.append(o_edge)
            
        for edge in edges_to_remove:
            if edge in self.inbound_edges:
                self.inbound_edges.remove(edge)
            if edge in edge.source.outbound_edges: 
                edge.source.outbound_edges.remove(edge)

    def adjust_lag(self, lags):
        z = self.point.get_z() * 10
        lag_step = 10/(lags)
        self.lag = round((z/lag_step))