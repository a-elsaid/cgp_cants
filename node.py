import numpy as np
from search_space import Point
from math import ceil, floor
from util import function_dict
import loguru

def sigmoid(x):
    # x = max(-100.0, min(100.0, x))
    return 1 / (1 + np.exp(-x))

logger = loguru.logger


class Edge():
    count = 0
    def __init__(self, source, target, weight=1.0):
        self.id = Edge.count + 1
        Edge.count += 1
        self.source = source
        self.target = target
        self.grad = 0.0
        if weight is None:
            self.weight = np.random.uniform(-0.5, 0.5)
        else:
            self.weight = weight


class Node():
    count = 0
    def __init__(
                    self, 
                    type=0, 
                    point: Point = None, 
                    lags=None,
    ):
        self.id = Node.count + 1
        Node.count += 1
        self.lag = None
        self.z = point.get_z()
        self.__cluster = None
        self.type = type                   # 0 = normal, 1 = input, 2 = output
        self.functions = {}
        self.backfire = 0.0
        self.forefire = []
        self.d_err = None                  # Error value for output nodes only
        self.node_value = 0.0
        self.recieved_fire = 0
        self.recieved_backfire = 0
        self.point = point
        self.inbound_edges = {}
        self.outbound_edges = {}
        self.active = False
        self.pick_node_functions(function_dict)
        self.adjust_lag(lags=lags-1)    # Adjust lag levels based on the z value of the point

    def get_cluseter(self):
        return self.__cluster

    def set_cluster(self, cluster):
        self.__cluster = cluster

    def add_functions(self, functions):
        self.functions.update(functions)

        
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
        elif int(func_id) == 0:
            self.functions.update({0: function_dict[0]})
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
        logger.debug(f"Adding edge from {self.id} to {to_node.id}")
        self.outbound_edges[to_node.id] = edge
        to_node.inbound_edges[self.id]  = edge


    def fire (self, ):
        results = []
        for func in self.functions.values():
            fn_res = np.clip(func(self.forefire), -5, 5)
            results.append(sigmoid(fn_res))

        result = np.average(results)
        result = np.clip(result, -5, 5)
        self.node_value = sigmoid(result)
        # self.node_value = np.sum(self.forefire)
        # self.node_value = sigmoid(np.sum(self.forefire))
        logger.debug(f"Node({self.id:5d}) is firing {self.node_value:.5f} [Signal({self.recieved_fire}/{len(self.inbound_edges)})]")
        for edge in self.outbound_edges.values():
            edge.target.recieve_fire(self.node_value * edge.weight)
        self.recieved_fire = 0
        self.forefire = []

    def recieve_fire (self, value):
        logger.debug(f"Node({self.id}) recieved fire {value}")
        self.recieved_fire+= 1
        self.forefire.append(value)
        if self.recieved_fire >= len(self.inbound_edges):
            self.fire()

    def update_weights(self, lr=0.001, m=1):
        for edge in self.inbound_edges.values():
            # edge.source.update_weights(edge.weight, lr=lr)
            edge.weight += lr * edge.grad
            edge.grad = 0.0
            
    def fireback (self,):
        for edge in self.inbound_edges.values():
            edge.source.recieve_backfire(self.backfire * self.node_value*(1-self.node_value) * edge.weight)
            edge.grad+= self.backfire * edge.source.node_value * self.node_value * (1 - self.node_value)
            edge.grad = np.clip(edge.grad, -0.5, 0.5)
        self.recieved_backfire = 0
        self.backfire = 0.0

    def recieve_backfire (self, value):
        self.recieved_backfire+= 1
        self.backfire+= value
        if self.recieved_backfire >= len(self.outbound_edges):
            self.fireback()

    def adjust_lag(self, lags):
        self.lag = round(self.point.get_z() * lags)
        self.z = self.lag / max(1,lags)