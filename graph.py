
from doctest import OutputChecker
import numpy as np
import op


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


def read_calculation_graph(model_dir):
    '''
    从model_dir中读取计算图
    '''
    if(model_dir[-1] != "/"):
        model_dir += "/"
    with open(model_dir + "graph.txt", "r") as f:
        graph_txt_content = f.readlines()

    graph = []

    for line in graph_txt_content:
        line = line.replace("\n", "").replace(" ", "")
        op_id = get_id(line)
        op_name = get_name(line)
        op_parameters = get_parameters(line)
        graph.append(
            name_op_map_dict[op_name](op_id, op_parameters, model_dir)
        )

    return graph


