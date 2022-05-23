

import numpy as np

class QInput():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.shape = [int(i) for i in parameters["shape"].replace("(", "")
            .replace(")", "").split(",")]
        self.dtype = parameters["dtype"]
        self.output_shape = self.shape
    
    def __str__(self):
        string = ""
        string += "%%%d=qinput(shape=%s, dtype=%s);"%(
            self.id, self.shape, self.dtype)
        return string


class Input():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.shape = [int(i) for i in parameters["shape"].replace("(", "")
            .replace(")", "").split(",")]
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=input(shape=%s);"%(
            self.id, self.shape)
        return string


class QConv2d():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_channel = int(parameters["output_channel"])
        self.input_channel = int(parameters["input_channel"])
        self.kernel_size = [int(i) for i in parameters["kernel_size"]
            .replace("(", "").replace(")", "").split(",")]
        self.weight = np.fromfile(model_dir + parameters["weight"], 
            dtype="uint8").reshape(
            self.output_channel, self.input_channel, 
            self.kernel_size[0], self.kernel_size[1]
        )
        self.bias = np.fromfile(model_dir + parameters["bias"], 
            dtype="int32").reshape(self.output_channel)
        self.stride = [int(i) for i in parameters["stride"].replace("(", "")
            .replace(")", "").split(",")]
        self.padding = [int(i) for i in parameters["padding"].replace("(", "")
            .replace(")", "").split(",")]
        self.dilation = [int(i) for i in parameters["dilation"]
            .replace("(", "").replace(")", "").split(",")]
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
        self.zero_x = int(parameters["zero_x"])
        self.zero_w = int(parameters["zero_w"])
        self.zero_b = int(parameters["zero_b"])
        self.zero_y = int(parameters["zero_y"])
        self.coe = float(parameters["coe"])
        self.rshift = int(parameters["rshift"])
        self.qmin = int(parameters["qmin"])
        self.qmax = int(parameters["qmax"])
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.qconv2d(input=%d, output_channel=%d, " \
            "input_channel=%d, kernel_size=%s, weight=%s, bias=%s, " \
            "stride=%s, padding=%s, dilation=%s, output_shape=%s, zero_x=%d, "\
            "zero_w=%d, zero_b=%d, zero_y=%d, coe=%f, rshift=%d, qmin=%d, " \
            "qmax=%d);"%(
                self.id, self.input, self.output_channel, self.input_channel, 
                self.kernel_size, self.weight.shape, self.bias.shape, 
                self.stride, self.padding, self.dilation, self.output_shape, 
                self.zero_x, self.zero_w, self.zero_b, self.zero_y, 
                self.coe, self.rshift, self.qmin, self.qmax
            )
        return string


class Conv2d():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_channel = int(parameters["output_channel"])
        self.input_channel = int(parameters["input_channel"])
        self.kernel_size = [int(i) for i in parameters["kernel_size"]
            .replace("(", "").replace(")", "").split(",")]
        self.weight = np.fromfile(model_dir + parameters["weight"],
            dtype="float32").reshape(
            self.output_channel, self.input_channel,
            self.kernel_size[0], self.kernel_size[1]
        )
        if(str(parameters["bias"]) == "None"):
            self.bias = np.zeros((self.output_channel))
        else:
            self.bias = np.fromfile(model_dir + parameters["bias"],
                dtype="float32").reshape(self.output_channel)
        self.stride = [int(i) for i in parameters["stride"].replace("(", "")
            .replace(")", "").split(",")]
        self.padding = [int(i) for i in parameters["padding"].replace("(", "")
            .replace(")", "").split(",")]
        self.dilation = [int(i) for i in parameters["dilation"]
            .replace("(", "").replace(")", "").split(",")]
        self.output_shape = None
        
    def __str__(self):
        string = ""
        string += "%%%d=nn.conv2d(input=%d, output_channel=%d, " \
            "input_channel=%d, kernel_size=%s, weight=%s, bias=%s, " \
            "stride=%s, padding=%s, dilation=%s, output_shape=%s);"%(
                self.id, self.input, self.output_channel, self.input_channel, 
                self.kernel_size, self.weight.shape, self.bias.shape, 
                self.stride, self.padding, self.dilation, self.output_shape
            )
        return string


class QMaxpool2d():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.kernel_size = [int(i) for i in parameters["kernel_size"]
            .replace("(", "").replace(")", "").split(",")]
        self.stride = [int(i) for i in parameters["stride"].replace("(", "")
            .replace(")", "").split(",")]
        self.padding = [int(i) for i in parameters["padding"].replace("(", "")
            .replace(")", "").split(",")]
        self.dilation = [int(i) for i in parameters["dilation"]
            .replace("(", "").replace(")", "").split(",")]
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
        self.zero = int(parameters["zero"])
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.qmaxpool2d(input=%d, kernel_size=%s, stride=%s, " \
            "padding=%s, dilation=%s, output_shape=%s, zero=%d);"%(
                self.id, self.input, self.kernel_size, self.stride, 
                self.padding, self.dilation, self.output_shape, self.zero
            )
        return string


class Maxpool2d():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.kernel_size = [int(i) for i in parameters["kernel_size"]
            .replace("(", "").replace(")", "").split(",")]
        self.stride = [int(i) for i in parameters["stride"].replace("(", "")
            .replace(")", "").split(",")]
        self.padding = [int(i) for i in parameters["padding"].replace("(", "")
            .replace(")", "").split(",")]
        self.dilation = [int(i) for i in parameters["dilation"]
            .replace("(", "").replace(")", "").split(",")]
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.maxpool2d(input=%d, kernel_size=%s, stride=%s, " \
            "padding=%s, dilation=%s, output_shape=%s);"%(
                self.id, self.input, self.kernel_size, self.stride, 
                self.padding, self.dilation, self.output_shape
            )
        return string


class QRelu():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
        self.zero = int(parameters["zero"])
        self.qmax = int(parameters["qmax"])
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.qrelu(input=%d, output_shape=%s, zero=%d, " \
            "qmax=%d);"%(
            self.id, self.input, self.output_shape, self.zero, self.qmax
        )
        return string


class Relu():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.relu(input=%d, output_shape=%s);"%(
            self.id, self.input, self.output_shape
        )
        return string


class QFlatten():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.qflatten(input=%d, output_shape=%s);"%(
            self.id, self.input, self.output_shape
        )
        return string


class Flatten():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.flatten(input=%d, output_shape=%s);"%(
            self.id, self.input, self.output_shape
        )
        return string


class QDense():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_channel = int(parameters["output_channel"])
        self.input_channel = int(parameters["input_channel"])
        self.weight = np.fromfile(model_dir + parameters["weight"], 
            "uint8").reshape(
            self.output_channel, self.input_channel
        )
        self.bias = np.fromfile(model_dir + parameters["bias"], 
            "int32").reshape(self.output_channel)
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
        self.zero_x = int(parameters["zero_x"])
        self.zero_w = int(parameters["zero_w"])
        self.zero_b = int(parameters["zero_b"])
        self.zero_y = int(parameters["zero_y"])
        self.coe = float(parameters["coe"])
        self.rshift = int(parameters["rshift"])
        self.qmin = int(parameters["qmin"])
        self.qmax = int(parameters["qmax"])

    def __str__(self):
        string = ""
        string += "%%%d=nn.qdense(input=%d, output_channel=%d, " \
            "input_channel=%d, weight=%s, bias=%s, output_shape=%s, " \
            "zero_x=%d, zero_w=%d, zero_b=%d, zero_y=%d, coe=%f, rshift=%d, " \
            "qmin=%d, qmax=%d);"%(
                self.id, self.input, self.output_channel, self.input_channel, 
                self.weight.shape, self.bias.shape, self.output_shape, 
                self.zero_x, self.zero_w, self.zero_b, self.zero_y,
                self.coe, self.rshift, self.qmin, self.qmax
            )
        return string


class Dense():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_channel = int(parameters["output_channel"])
        self.input_channel = int(parameters["input_channel"])
        self.weight = np.fromfile(model_dir + parameters["weight"], 
            "float32").reshape(
            self.output_channel, self.input_channel
        )
        self.bias = np.fromfile(model_dir + parameters["bias"], 
            "float32").reshape(self.output_channel)
        self.output_shape = None

    def __str__(self):
        string = ""
        string += "%%%d=nn.dense(input=%d, output_channel=%d, " \
            "input_channel=%d, weight=%s, bias=%s, output_shape=%s);"%(
                self.id, self.input, self.output_channel, self.input_channel, 
                self.weight.shape, self.bias.shape, self.output_shape
            )
        return string


class QDropout():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.p = float(parameters["p"])
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.qdropout(input=%d, p=%f, output_shape=%s);"%(
            self.id, self.input, self.p, self.output_shape
        )
        return string


class Dropout():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.p = float(parameters["p"])
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.dropout(input=%d, p=%f, output_shape=%s);"%(
            self.id, self.input, self.p, self.output_shape
        )
        return string


class QOutput():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
    
    def __str__(self):
        string = ""
        string += "%%%d=qoutput(input=%d, output_shape=%s);"%(
            self.id, self.input, self.output_shape
        )
        return string


class Output():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=output(input=%d, output_shape=%s);"%(
            self.id, self.input, self.output_shape
        )
        return string


class QAdd():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input1 = int(parameters["input1"].replace("%", ""))
        self.input2 = int(parameters["input2"].replace("%", ""))
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
        self.zero_x1 = int(parameters["zero_x1"])
        self.zero_x2 = int(parameters["zero_x2"])
        self.zero_y = int(parameters["zero_y"])
        self.coe1 = float(parameters["coe1"])
        self.coe2 = float(parameters["coe2"])
        self.rshift1 = int(parameters["rshift1"])
        self.rshift2 = int(parameters["rshift2"])
        self.qmin = int(parameters["qmin"])
        self.qmax = int(parameters["qmax"])
    
    def __str__(self):
        string = ""
        string += "%%%d=qadd(input1=%d, input2=%d, output_shape=%s, " \
            "zero_x1=%d, zero_x2=%d, zero_y=%d, coe1=%f, coe2=%f, " \
            "rshift1=%d, rshift2=%d, qmin=%d, qmax=%d);"%(
                self.id, self.input1, self.input2, self.output_shape, 
                self.zero_x1, self.zero_x2, self.zero_y, self.coe1, self.coe2,
                self.rshift1, self.rshift2, self.qmin, self.qmax
            )
        return string


class Add():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input1 = int(parameters["input1"].replace("%", ""))
        self.input2 = int(parameters["input2"].replace("%", ""))
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=add(input1=%d, input2=%d, output_shape=%s);"%(
                self.id, self.input1, self.input2, self.output_shape
            )
        return string


class QConcat():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input1 = int(parameters["input1"].replace("%", ""))
        self.input2 = int(parameters["input2"].replace("%", ""))
        self.dim = int(parameters["dim"])
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
        self.zero_x1 = int(parameters["zero_x1"])
        self.zero_x2 = int(parameters["zero_x2"])
        self.zero_y = int(parameters["zero_y"])
        self.coe1 = float(parameters["coe1"])
        self.coe2 = float(parameters["coe2"])
        self.rshift1 = int(parameters["rshift1"])
        self.rshift2 = int(parameters["rshift2"])
        self.qmin = int(parameters["qmin"])
        self.qmax = int(parameters["qmax"])
    
    def __str__(self):
        string = ""
        string += "%%%d=qconcat(input1=%d, input2=%d, dim=%d, " \
            "output_shape=%d, zero_x1=%d, zero_x2=%d, zero_y=%d, coe1=%f, " \
            "coe2=%f, rshift1=%d, rshift2=%d, qmin=%d, qmax=%d);"%(
                self.id, self.input1, self.input2, self.dim, self.output_shape,
                self.zero_x1, self.zero_x2, self.zero_y, self.coe1, self.coe2,
                self.rshift1, self.rshift2, self.qmin, self.qmax
            )
        return string


class Concat():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input1 = int(parameters["input1"].replace("%", ""))
        self.input2 = int(parameters["input2"].replace("%", ""))
        self.dim = int(parameters["dim"])
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=concat(input1=%d, input2=%d, dim=%d, " \
            "output_shape=%s);"%(
                self.id, self.input1, self.input2, self.dim, self.output_shape
            )
        return string


class QAvgpool2d():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.kernel_size = [int(i) for i in parameters["kernel_size"]
            .replace("(", "").replace(")", "").split(",")]
        self.stride = [int(i) for i in parameters["stride"].replace("(", "")
            .replace(")", "").split(",")]
        self.padding = [int(i) for i in parameters["padding"].replace("(", "")
            .replace(")", "").split(",")]
        self.output_shape = [int(i) for i in parameters["output_shape"]
            .replace("(", "").replace(")", "").split(",")]
        self.zero = int(parameters["zero"])
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.qavgpool2d(input=%d, kernel_size=%s, stride=%s, " \
            "padding=%s, output_shape=%s, zero=%d);"%(
            self.id, self.input, self.kernel_size, self.stride, self.padding, 
            self.output_shape, self.zero
        )
        return string


class Avgpool2d():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.kernel_size = [int(i) for i in parameters["kernel_size"]
            .replace("(", "").replace(")", "").split(",")]
        self.stride = [int(i) for i in parameters["stride"].replace("(", "")
            .replace(")", "").split(",")]
        self.padding = [int(i) for i in parameters["padding"].replace("(", "")
            .replace(")", "").split(",")]
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.avgpool2d(input=%d, kernel_size=%s, stride=%s, " \
            "padding=%s, output_shape=%s);"%(
            self.id, self.input, self.kernel_size, self.stride, self.padding,
            self.output_shape
        )
        return string


class Batch_norm2d():
    def __init__(
        self,
        id: int,
        parameters: dict,
        model_dir: str
    ):
        self.id = id
        self.input = int(parameters["input"].replace("%", ""))
        self.num_features = int(parameters["num_features"])
        self.weight = np.fromfile(model_dir + parameters["weight"], 
            "float32").reshape(self.num_features)
        self.bias = np.fromfile(model_dir + parameters["bias"],
            "float32").reshape(self.num_features)
        self.running_mean = np.fromfile(model_dir + parameters["running_mean"],
            "float32").reshape(self.num_features)
        self.running_var = np.fromfile(model_dir + parameters["running_var"],
            "float32").reshape(self.num_features)
        self.eps = float(parameters["eps"])
        self.momentum = float(parameters["momentum"])
        self.output_shape = None
    
    def __str__(self):
        string = ""
        string += "%%%d=nn.batch_norm2d(input=%d, weight=%s, bias=%s, " \
            "running_mean=%s, running_var=%s, num_features=%d, eps=%f, " \
            "momentum=%f, output_shape=%s);"%(
            self.id, self.input, self.weight.shape, self.bias.shape, 
            self.running_mean.shape, self.running_var.shape, self.num_features,
            self.eps, self.momentum, self.output_shape
        )
        return string