import numpy as np
from search_space import Point
from math import ceil, floor
from util import function_dict, function_names
import loguru
import torch
import random

def sigmoid(x):
    # x = max(-100.0, min(100.0, x))
    return 1 / (1 + np.exp(-x))

logger = loguru.logger


class Edge():
    count = 0
    def __init__(self, source, target, weight=1.0, use_torch=False):
        self.id = Edge.count + 1
        Edge.count += 1
        self.source = source
        self.target = target
        self.grad = 0.0
        self.velocity = torch.tensor(0.0, dtype=torch.float64, requires_grad=True) if use_torch else 0.0

        if use_torch:
            self.weight = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
            # self.weight = torch.rand(1, dtype=torch.float64, requires_grad=True)
        elif weight is None:
            self.weight = np.random.uniform(-0.5, 0.5)
        else:
            self.weight = 1.0


class Node():
    count = 0
    def __init__(
                    self, 
                    type=0, 
                    point: Point = None, 
                    lags=None,
                    use_torch=True,
    ):
        self.id = Node.count + 1
        self.__use_torch = use_torch
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
        if self.type == 2:
            self.functions = {0: function_dict[0]}
        else:
            self.pick_node_functions(function_dict)
        self.adjust_lag(lags=lags-1)    # Adjust lag levels based on the z value of the point

    def get_cluseter(self):
        return self.__cluster

    def set_cluster(self, cluster):
        self.__cluster = cluster

    def add_functions(self, functions):
        if self.type == 2:
            self.functions = {0: function_dict[0]}
        else:
            '''Add the functions to the node'''
            # self.functions.update(functions)

            '''Ramdomly pick a function and make it the only function for the node'''
            random_key = np.random.choice(list(functions.keys()))
            self.functions = {random_key: function_dict[random_key]}

        
    def compare_corr(self, node):
        return  (
                    self.point.get_x() == node.point.get_x() and 
                    self.point.get_y() == node.point.get_y() and 
                    self.point.get_z() == node.point.get_z()
        )

    
    def pick_node_functions(self, function_dict):
        func_coord = self.point.get_f()
        func_id = (len(function_dict) - 1) * func_coord

        if int(func_id) == func_id:
            func_id = int(func_id)
            self.functions.update({func_id: function_dict[func_id]})
        elif int(func_id) == 0:
            self.functions.update({0: function_dict[0]})
        else:
            prev_func_id = int(func_id - 1)
            next_func_id = int(func_id + 1)
            # print(prev_func_id, next_func_id)   
            self.functions.update(
                                    {
                                        prev_func_id: function_dict[prev_func_id], 
                                        next_func_id: function_dict[next_func_id],
                                    }
            )
        random_id = random.choice(list(self.functions.keys()))
        self.functions = {random_id: self.functions[random_id]}


    def add_edge(self, to_node, weight=1.0):
        edge = Edge(self, to_node, use_torch=self.__use_torch, weight=weight)
        if to_node.id == self.id:
            return
        logger.debug(f"Adding edge from {self.id} to {to_node.id}")
        self.outbound_edges[to_node.id] = edge
        to_node.inbound_edges[self.id]  = edge


    def fire (self, ):
        results = []
        for fn_id, func in self.functions.items():
            if self.__use_torch:
                # fn_res = torch.clamp(func(torch.stack(self.forefire)), min=-1, max=1)
                fn_res = func(torch.stack(self.forefire))
            else:
                # fn_res = np.clip(func(self.forefire), -3.1, 3.1)
                fn_res = func(self.forefire)
            results.append(fn_res)
            self.node_value = torch.mean(torch.stack(results)) if self.__use_torch else np.mean(results)
            
        logger.debug(f"Node({self.id:5d}) is firing {self.node_value:.5f}")

        
        for edge in self.outbound_edges.values():
            logger.debug(f"Edge ID({edge.id}) Node({edge.source.id})->Node({edge.target.id}) Weight({edge.weight})  ID:({id(edge.weight)})")
            node_value = self.node_value * edge.weight
            node_value.retain_grad()
            edge.target.recieve_fire(node_value)
        self.recieved_fire = 0
        self.forefire = []

    def recieve_fire (self, value):
        self.recieved_fire+= 1
        logger.debug(f"Node({self.id}) recieved fire {value}   [Signal({self.recieved_fire}/{len(self.inbound_edges)})]")
        self.forefire.append(value)
        if self.recieved_fire >= len(self.inbound_edges):
            self.fire()

    def update_weights(self, lr=0.001, momentum=0.1):
        def update_weights():
            for edge in self.inbound_edges.values():
                edge.source.update_weights()
                edge.velocity = momentum * edge.velocity + (edge.weight.grad if self.__use_torch else edge.grad)
                edge.weight -= torch.clamp(lr * edge.velocity, min=-10, max=10) if self.__use_torch else lr * edge.grad
                edge.grad = 0.0
                if self.__use_torch:
                    edge.weight.grad.zero_()

        if self.__use_torch:
            with torch.no_grad():
                update_weights()
        else:
            update_weights()
            
            
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


    def get_eqn(self):
        eqn = ""
        
        node_funs = list(self.functions.keys())
        operator = "+"
        if function_names[node_funs[0]] == "multiply":
            operator = "*"    

        if self.type == 1:
            return f"{self.point.name} "

        for edge in self.inbound_edges.values():
            eqn += f"{edge.source.get_eqn()} * {edge.weight} {operator} "
        
        eqn = f"({eqn[:-3]})"
        if function_names[node_funs[0]] not in ['multiply', 'add']:
            eqn = f"{node_funs[0]}{eqn}"

        return eqn