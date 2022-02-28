# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import math
import typing
import json
import copy
from icecream import ic

DEBUG = True

class LayerType:
  INPUT = "input"
  HIDDEN = "hidden"
  OUTPUT = "output"

class ActivationFunction:
  SIGMOID = "sigmoid"
  RELU = "relu"
  SOFTMAX = "softmax"
  LINEAR = "linear"

# UTILS
class Utils:
  @staticmethod
  def matrix_dimension(mat: typing.List[list]) -> typing.Tuple[int,int]:
    return len(mat), len(mat[0])

  # @staticmethod
  # def export_txt(filename, data):
  # 	with open(filename, 'w') as f:
  # 		pass

  # @staticmethod
  # def parse_txt(filename):
  # 	with open(filename, 'r') as f:
  # 		lines = f.readlines()

  # 		count = 0
  # 		result = []
  # 		for line in lines:
  # 			result = line.split(" ")
  # 			count += 1
  # 		return result

  @staticmethod
  def parse_json(filename):
    with open(filename, 'r') as f:
      data = json.load(f)
      return data

  @staticmethod
  def export_json(filename, data):
    with open(filename, 'w') as f:
      f.write(json.dumps(data))

class Layer:
  def __init__(self, weights: typing.List[list],bias_weights: typing.List[float], values: typing.List[float],layer_type: LayerType, label: str, num_nodes:int,activation_func: ActivationFunction):
    self.weights = weights
    self.bias_weights = bias_weights
    self.values = values
    self.label = label
    self.type = layer_type
    self.num_nodes = num_nodes
    self.activation_func = activation_func

  def __str__(self):
    return "{} layer with {} weights and {} values".format(self.label,\
      len(self.weights), len(self.values))


class Graph:
  def __init__(self, layers : typing.List[Layer]):
    self.layers = layers

  def __str__(self):
    return "Graph with {} layers".format(len(self.layers))

  def create_empty_layer(self,prev_layer:Layer,num_nodes:int,layer_type:LayerType,label:str,activation_func:ActivationFunction)->Layer:
    weights = [[0 for i in range(num_nodes)] for j in range(prev_layer.num_nodes)]
    bias_weights = [0 for i in range(num_nodes)]
    values = [0 for i in range(num_nodes)]
    return Layer(weights,bias_weights,values,layer_type,label,num_nodes,activation_func)

  def create_input_layer(self,values:typing.List[float],label:str)->Layer:
    return Layer(None,None,values,LayerType.INPUT,label,len(values),None)

  def add_layer(self, layer:Layer):
    self.layers.append(layer)

  def net_value(self, layer_idx:int, node_idx:int):
    # get the net value of the <node_idx>th node in <layer_idx>th layer 
    rows = Utils.matrix_dimension(self.layers[layer_idx].weights)[0]

    if self.layers[layer_idx].type == LayerType.INPUT:
      return None

    val = self.layers[layer_idx].bias_weights[node_idx]
    for i in range(rows):
       val += self.layers[layer_idx].weights[i][node_idx] * self.layers[layer_idx-1].values[i]

    return val

  def layer_value(self,layer_idx:int):
    layer = self.layers[layer_idx]
    if layer.activation_func == ActivationFunction.LINEAR:
      pass
    elif layer.activation_func == ActivationFunction.SIGMOID:
      pass
    elif layer.activation_func == ActivationFunction.RELU:
      for i in range(len(layer.values)):
        layer.values[i] = self.relu_activation(layer_idx, i)
    elif layer.activation_func == ActivationFunction.SOFTMAX:
      for i in range(len(layer.values)):
        layer.values[i] = self.softmax_activation(layer_idx, i)

  def linear_activation(self):
    pass

  def sigmoid_activation(self):
    pass

  def relu_activation(self, layer_idx:int, node_idx:int):
    return max(0, self.net_value(layer_idx,node_idx))

  def softmax_activation(self, layer_idx:int, node_idx:int):
    sum_exp = 0
    for i in range(self.layers[layer_idx].num_nodes):
      sum_exp+=math.exp(self.net_value(layer_idx,i))

    return math.exp(self.net_value(layer_idx,node_idx))/sum_exp

    
# graph = Graph([])

# graph.add_layer(graph.input_layer([1,1],'input'))
# el = Layer([[20,-20],[20,-20]],[-10,30],[0,0],LayerType.HIDDEN,'hidden1',2,ActivationFunction.RELU)
# graph.add_layer(el)
# graph.layer_value(1)
# print(graph.layers[1].values)

# dim = Utils.matrix_dimension(el.weights)


if (DEBUG):
  filename = "data/ppt_example.json"
  
  data = Utils.parse_json(filename)
  layer1_matrix = data["layers"][1]['values']
  ic(data)
  ic(layer1_matrix)
  
  new_filename = "data/ppt_example_out.json"
  new_data = copy.deepcopy(data)
  new_data["layers"][1]['values'] = [[1,2], [3,9999]]
  Utils.export_json(new_filename, new_data)
  
  ic(data)
  ic(new_data)
  assert data is not new_data, print("harusnya beda")