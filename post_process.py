
from socketserver import DatagramRequestHandler
import numpy as np
import math


def coe_to_bin(data):
    bin = ""
    temp = 0.0
    for i in range(16):
        if(temp + 2**(-1-i) > data):
            bin += "0"
        else:
            bin += "1"
            temp += 2**(-1-i)
    return bin


def decimal_to_binary(decimal_number, width):
    if(decimal_number < 0):
        raise ValueError("decimal_number must be positive")
    bin = ""
    if(decimal_number == 0):
        for i in range(width):
            bin += "0"
    while(decimal_number > 0):
        bin = str(decimal_number % 2) + bin
        decimal_number //= 2
    while(len(bin) < width):
        bin = "0" + bin
    return bin


def post_process(
    MODULE_NAME: str,                   # 模块名
    MUX_WIDTH: int,                     # mux位宽
    DATA_WIDTH: int,                    # 输入位宽(must be 32)
    DATA_NUMBER: int,                   # 并行输入数据数量
    OUT_DATA_WIDTH: int,                # 输出位宽
    BIAS: list,                         # bias
    COE: list,                          # coe
    RSHIFT: list,                       # rshift
    ZERO_Y: list,                       # zero_y
    QMAX: int,                          # qmax
    DEBUG=True                          # debug
):
    # 需要添加DEBUG信息的信号列表。每个元素是一个列表，里面有3个值，分别是信号名称，信号位宽，信号数组深度
    debug_signals = []
    # 中间结果各级寄存器数量
    intermediate_reg_number = {}

    # 最后要返回的code
    code = ""

    # 生成模块
    code += "module %s(\n"%(MODULE_NAME)
    indent = "\t"
    code += indent + "input clk,\n"
    code += indent + "input [%d:0] in_contiguous,\n"%(DATA_WIDTH*DATA_NUMBER-1)
    max_bias_len = 0
    for b in BIAS:
        if(len(b) > max_bias_len):
            max_bias_len = len(b)
    channel_width = 0
    while(2**channel_width < max_bias_len):
        channel_width += 1
    channel_width = int(math.ceil(channel_width/8))*8
    code += indent + "input [%d:0] channel,\n"%(channel_width-1)
    code += indent + "input [%d:0] mux,\n"%(MUX_WIDTH-1)
    code += indent + "input do_relu,\n"
    code += indent + "output [%d:0] out_contiguous,\n"%(OUT_DATA_WIDTH*DATA_NUMBER-1)
    code = code[:-2] + "\n"
    code += ");\n"

    # 生成coe
    for n, c in enumerate(COE):
        code += indent + "parameter [47:0] coe%d = 48'b%s;\n"%(n, coe_to_bin(c))
    
    # 生成rshift
    for n, r in enumerate(RSHIFT):
        code += indent + "parameter [4:0] rshift%d = %d;\n"%(n, r)

    # 生成zero_y
    for n, z in enumerate(ZERO_Y):
        code += indent + "parameter [31:0] zero_y%d = %d;\n"%(n, z)

    # 生成bias
    for n, b in enumerate(BIAS):
        code += indent + "reg [31:0] bias%d[%d:0];\n"%(n, len(b)-1)
    
    # 生成qmax
    code += indent + "parameter [31:0] qmax = %d;\n"%(QMAX)

    # 生成coe
    code += indent + "reg [47:0] coe;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for n, c in enumerate(COE):
        indent = "\t\t\t"
        code += indent + "%d'b%s: coe = coe%d;\n"%(MUX_WIDTH, decimal_to_binary(n, MUX_WIDTH), n)
    code += indent + "default: coe = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成rshift
    code += indent + "reg [4:0] rshift;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for n, r in enumerate(RSHIFT):
        indent = "\t\t\t"
        code += indent + "%d'b%s: rshift = rshift%d;\n"%(MUX_WIDTH, decimal_to_binary(n, MUX_WIDTH), n)
    code += indent + "default: rshift = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成zero_y
    code += indent + "reg [31:0] zero_y;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for n, z in enumerate(ZERO_Y):
        indent = "\t\t\t"
        code += indent + "%d'b%s: zero_y = zero_y%d;\n"%(MUX_WIDTH, decimal_to_binary(n, MUX_WIDTH), n)
    code += indent + "default: zero_y = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成bias
    code += indent + "reg [31:0] bias;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for n, b in enumerate(BIAS):
        indent = "\t\t\t"
        code += indent + "%d'b%s: bias = bias%d[channel];\n"%(MUX_WIDTH, decimal_to_binary(n, MUX_WIDTH), n)
    code += indent + "default: bias = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成in
    code += indent + "wire signed [31:0] in[%d:0];\n"%(DATA_NUMBER-1)
    debug_signals.append(["in", 32, DATA_NUMBER])
    for i in range(DATA_NUMBER):
        code += indent + "assign in[%d] = in_contiguous[%d:%d];\n"%(i, i*DATA_WIDTH+DATA_WIDTH-1, i*DATA_WIDTH)
    
    # 生成in_add_bias
    code += indent + "reg signed [31:0] in_add_bias[%d:0];\n"%(DATA_NUMBER-1)
    debug_signals.append(["in_add_bias", 32, DATA_NUMBER])

    # 生成fp_temp
    code += indent + "reg fp_temp_sign[%d:0];\n"%(DATA_NUMBER-1)
    code += indent + "reg [47:0] fp_temp[%d:0];\n"%(DATA_NUMBER-1)
    debug_signals.append(["fp_temp_sign", 1, DATA_NUMBER])
    debug_signals.append(["fp_temp", 48, DATA_NUMBER])

    # 生成mult_temp
    code += indent + "wire [95:0] mult_temp[%d:0];\n"%(DATA_NUMBER-1)
    debug_signals.append(["mult_temp", 96, DATA_NUMBER])
    for i in range(DATA_NUMBER):
        code += indent + "assign mult_temp[%d] = fp_temp[%d] * coe;\n"%(i, i)

    # 生成mult
    code += indent + "reg mult_sign[%d:0];\n"%(DATA_NUMBER-1)
    code += indent + "reg [47:0] mult[%d:0];\n"%(DATA_NUMBER-1)
    debug_signals.append(["mult_sign", 1, DATA_NUMBER])
    debug_signals.append(["mult", 48, DATA_NUMBER])

    # 生成t
    code += indent + "reg signed [31:0] t[%d:0];\n"%(DATA_NUMBER-1)
    debug_signals.append(["t", 32, DATA_NUMBER])

    # 生成t_add
    code += indent + "reg signed [31:0] t_add[%d:0];\n"%(DATA_NUMBER-1)
    debug_signals.append(["t_add", 32, DATA_NUMBER])

    # 生成relu
    code += indent + "reg signed [31:0] relu[%d:0];\n"%(DATA_NUMBER-1)
    debug_signals.append(["relu", 32, DATA_NUMBER])
    for i in range(DATA_NUMBER):
        code += indent + "assign out_contiguous[%d:%d] = relu[%d][%d:0];\n"%(
            i*OUT_DATA_WIDTH+OUT_DATA_WIDTH-1, i*OUT_DATA_WIDTH, i, OUT_DATA_WIDTH-1
        )
    
    # 生成流水计算模块
    code += indent + "always @(posedge clk) begin\n"
    indent = "\t\t"
    #   # temp = matmul_res + bias[channel] - zero_b
    for i in range(DATA_NUMBER):
        code += indent + "in_add_bias[%d] <= in[%d] + bias;\n"%(i, i)
    #   # fp_temp = temp
    for i in range(DATA_NUMBER):
        code += indent + "fp_temp_sign[%d] <= in_add_bias[%d][31];\n"%(i, i)
        code += indent + "fp_temp[%d] <= {{(in_add_bias[%d] - in_add_bias[%d][31]) ^ " \
        "({32{in_add_bias[%d][31]}})}, 16'b0};\n"%(i, i, i, i)
    #   # fp_temp *= coe
    for i in range(DATA_NUMBER):
        code += indent + "mult_sign[%d] <= fp_temp_sign[%d];\n"%(i, i)
        code += indent + "mult[%d] <= mult_temp[%d][63:16];\n"%(i, i)
    #   # t = fp_temp->to_int() >> rshift
    for i in range(DATA_NUMBER):
        code += indent + "t[%d] <= ((mult[%d][47:16] >> rshift) ^ {32{mult_sign[%d]}}) + mult_sign[%d];\n"%(i, i, i, i)
    #   # t_add = t + zero_y
    for i in range(DATA_NUMBER):
        code += indent + "t_add[%d] <= t[%d] + zero_y;\n"%(i, i)
    #   # relu
    code += indent + "if(do_relu) begin\n"
    indent = "\t\t\t"
    for i in range(DATA_NUMBER):
        code += indent + "relu[%d] <= (t_add[%d] < zero_y) ? zero_y : \n"%(i, i)
        code += indent + "            (t_add[%d] > qmax) ? qmax : \n"%(i)
        code += indent + "            t_add[%d];\n"%(i)
    indent = "\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t"
    for i in range(DATA_NUMBER):
        code += indent + "relu[%d] <= (t_add[%d] < 0) ? 0 : \n"%(i, i)
        code += indent + "            (t_add[%d] > qmax) ? qmax : \n"%(i)
        code += indent + "            t_add[%d];\n"%(i)
    indent = "\t\t"
    code += indent + "end\n"
    indent = "\t"
    code += indent + "end\n"

    # initial
    code += indent + "initial begin\n"
    indent = "\t\t"
    for n, b in enumerate(BIAS):
        for ni, i in enumerate(b):
            code += indent + "bias%d[%d] = %d;\n"%(n, ni, i)

    indent = "\t"
    code += indent + "end\n"


    code += "endmodule\n"
    return code