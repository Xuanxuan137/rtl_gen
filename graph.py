

import numpy as np
import op
from util import *


name_op_map_dict = {
    "qinput": op.QInput,
    "nn.qconv2d": op.QConv2d,
    "nn.qmaxpool2d": op.QMaxpool2d,
    "nn.qrelu": op.QRelu,
    "nn.qflatten": op.QFlatten,
    "nn.qdense": op.QDense,
    "nn.qdropout": op.QDropout,
    "qoutput": op.QOutput,
    "qadd": op.QAdd,
    "qconcat": op.QConcat,
    "nn.qavgpool2d": op.QAvgpool2d,

    "input": op.Input,
    "nn.conv2d": op.Conv2d,
    "nn.maxpool2d": op.Maxpool2d,
    "nn.relu": op.Relu,
    "nn.flatten": op.Flatten,
    "nn.dense": op.Dense,
    "nn.dropout": op.Dropout,
    "output": op.Output,
    "add": op.Add,
    "concat": op.Concat,
    "nn.avgpool2d": op.Avgpool2d,
    "nn.batch_norm2d": op.Batch_norm2d,
}


def get_id(line):
    '''
    从graph.txt的一行中提取算子id
    '''
    op_id = ""
    n = 1
    while(True):
        if(line[n] == "="):
            break
        op_id += line[n]
        n += 1
    op_id = int(op_id)
    return op_id


def get_name(line):
    '''
    从graph.txt的一行中提取算子名称
    '''
    op_name = ""
    n = 0
    while(True):
        if(line[n] == "="):
            break
        n += 1
    n += 1
    while(True):
        if(line[n] == "("):
            break
        op_name += line[n]
        n += 1
    return op_name


def get_parameters(line):
    '''
    从graph.txt的一行中提取算子参数
    '''
    parameters = {}
    n = 0
    while(True):
        if(line[n] == "("):
            break
        n += 1
    n += 1
    parameter = ""
    bracket = 0
    while(True):
        if(line[n] == "("):
            bracket += 1
        if(line[n] == ")"):
            bracket -= 1
        if(bracket == -1):
            parameter_name = parameter.split("=")[0]
            parameter_value = parameter.split("=")[1]
            parameter = ""
            parameters[parameter_name] = parameter_value
            break
        if(line[n] == "," and bracket == 0):
            parameter_name = parameter.split("=")[0]
            parameter_value = parameter.split("=")[1]
            parameter = ""
            parameters[parameter_name] = parameter_value
        else:
            parameter += line[n]
        n += 1
    return parameters


def infer_output_shape(calculation_graph):
    '''
    推断计算图中的output_shape
    '''
    for n, node in enumerate(calculation_graph):
        if(not node.output_shape is None):
            continue

        if(type(node) == op.Add or 
            type(node) == op.Concat):
            input1_shape = calculation_graph[node.input1].output_shape
            input2_shape = calculation_graph[node.input2].output_shape
            # TODO: 这里只考虑了input_1=input2的情况
            calculation_graph[n].output_shape = input1_shape
        elif(type(node) == op.Input):
            calculation_graph[n].output_shape = calculation_graph[n].shape
        else:
            input_shape = calculation_graph[node.input].output_shape
            if(type(node) == op.Conv2d):
                batch_size = input_shape[0]
                channel = node.weight.shape[0]
                padded_h = input_shape[2] + node.padding[0] * 2
                padded_w = input_shape[3] + node.padding[1] * 2
                height = (padded_h - (node.dilation[0] * (node.kernel_size[0]
                    -1) + 1)) // node.stride[0] + 1
                width = (padded_w - (node.dilation[1] * (node.kernel_size[1]
                    -1) + 1)) // node.stride[1] + 1
                calculation_graph[n].output_shape = [
                    batch_size, channel, height, width]
            elif(type(node) == op.Maxpool2d):
                batch_size = input_shape[0]
                channel = input_shape[1]
                padded_h = input_shape[2] + node.padding[0] * 2
                padded_w = input_shape[3] + node.padding[1] * 2
                height = (padded_h - (node.dilation[0] * (node.kernel_size[0]
                    -1) + 1)) // node.stride[0] + 1
                width = (padded_w - (node.dilation[1] * (node.kernel_size[1]
                    -1) + 1)) // node.stride[1] + 1
                calculation_graph[n].output_shape = [
                    batch_size, channel, height, width]
            elif(type(node) == op.Avgpool2d):
                batch_size = input_shape[0]
                channel = input_shape[1]
                padded_h = input_shape[2] + node.padding[0] * 2
                padded_w = input_shape[3] + node.padding[1] * 2
                height = (padded_h - node.kernel_size[0]) // node.stride[0] + 1
                width = (padded_w - node.kernel_size[1]) // node.stride[1] + 1
                calculation_graph[n].output_shape = [batch_size, channel, height, width]
            elif(type(node) == op.Dense):
                batch_size = input_shape[0]
                output_channel = node.weight.shape[0]
                calculation_graph[n].output_shape = [batch_size, output_channel]
            elif(type(node) == op.Flatten):
                batch_size = input_shape[0]
                output_channel = 1
                for i in input_shape[1:]:
                    output_channel *= i
                calculation_graph[n].output_shape = [batch_size, output_channel]
            elif(type(node) == op.Relu or
                 type(node) == op.Dropout or
                 type(node) == op.Output or
                 type(node) == op.Batch_norm2d):
                calculation_graph[n].output_shape = input_shape
            else:
                raise TypeError("Unknown op")
    


def read_calculation_graph(model_dir):
    '''
    从model_dir中读取计算图
    '''
    if(model_dir[-1] != "/"):
        model_dir += "/"
    with open(model_dir + "graph.txt", "r") as f:
        graph_txt_content = f.readlines()

    graph = []

    xxlog("Reading calculation graph")
    for line in graph_txt_content:
        line = line.replace("\n", "").replace(" ", "")
        op_id = get_id(line)
        op_name = get_name(line)
        op_parameters = get_parameters(line)
        graph.append(
            name_op_map_dict[op_name](op_id, op_parameters, model_dir)
        )
        xxlog("Read op %%%s=%s from graph.txt"%(op_id, op_name))
    xxlog("Read calculation graph finished")

    return graph


