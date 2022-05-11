
from audioop import bias
import math
import numpy as np



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


def gen_fc(
    MODULE_NAME: str,                   # 模块名
    MUX_WIDTH: int,                     # mux位宽
    DATA_WIDTH: int,                    # 数据位宽
    DATA_NUMBER: int,                   # 并行输入数据数量
    HIDDEN_LEN: int,                    # 上一层节点数量
    OUTPUT_LEN: int,                    # 本层节点数量
    BIAS: list,                         # 偏置。list里存各层bias的np.ndarray
    COE: list,                          # coe
    RSHIFT: list,                       # rshift
    ZERO_X: list,                       # zero_x
    ZERO_W: list,                       # zero_w
    ZERO_Y: list,                       # zero_y
    QMAX: int,                          # qmax
    DEBUG=True,                         # 是否打印调试信息
):

    # 需要添加DEBUG信息的信号列表。每个元素是一个列表，里面有3个值，分别是信号名称，
    # 信号位宽，信号数组深度
    debug_signals = []
    # 中间结果各级寄存器数量
    intermediate_reg_number = {}

    # 最后要返回的code
    code = ""

    # 生成模块
    code += "module %s(\n"%(MODULE_NAME)

    # 生成端口
    indent = "\t"
    code += indent + "input clk,\n"
    code += indent + "input [%d:0] din,\n"%(DATA_WIDTH*DATA_NUMBER-1)
    code += indent + "input in_valid,\n"
    code += indent + "input data_type,\n"
    len_number = int(math.ceil(HIDDEN_LEN / 8))     # 隐藏层节点数/8
    n = 0
    while(2**n < len_number):
        n += 1
    n = int(math.ceil(n/8))*8
    code += indent + "input [%d:0] len,\n"%(n-1)
    n = 0
    while(2**n < OUTPUT_LEN):
        n += 1
    n = int(math.ceil(n/8))*8
    code += indent + "input [%d:0] out_len,\n"%(n-1)
    code += indent + "input [%d:0] mux,\n"%(MUX_WIDTH-1)
    code += indent + "input do_relu,\n"
    code += indent + "output reg [%d:0] dout,\n"%(DATA_WIDTH-1)
    code += indent + "output reg out_valid,\n"
    code = code[:-2] + "\n" # 去除最后一个逗号
    code += ");\n"

    # 生成bias
    for n, b in enumerate(BIAS):
        code += indent + "reg [31:0] bias%d[%d:0];\n"%(n, len(b)-1)

    # 生成coe
    for n, c in enumerate(COE):
        code += indent + "parameter [47:0] coe%d = 48'b%s;\n"%(
            n, coe_to_bin(c))
    
    # 生成rshift
    for n, r in enumerate(RSHIFT):
        code += indent + "parameter [4:0] rshift%d = %d;\n"%(n, r)
    
    # 生成zero_x
    for n, x in enumerate(ZERO_X):
        code += indent + "parameter signed [%d:0] zero_x%d = %d;\n"%(
            DATA_WIDTH, n, x)
    
    # 生成zero_w
    for n, w in enumerate(ZERO_W):
        code += indent + "parameter signed [%d:0] zero_w%d = %d;\n"%(
            DATA_WIDTH, n, w)
    
    # 生成zero_y
    for n, y in enumerate(ZERO_Y):
        code += indent + "parameter signed [31:0] zero_y%d = %d;\n"%(
            DATA_WIDTH, y)
    
    # 生成qmax
    code += indent + "parameter signed [31:0] qmax = %d;\n"%(QMAX)

    # 生成输入层数据暂存空间data
    code += indent + "reg [%d:0] data[%d:0];\n"%(
        DATA_WIDTH*DATA_NUMBER-1, int(math.ceil(HIDDEN_LEN/8))-1)

    # 生成data地址信号addr
    n = 0
    len_number = int(math.ceil(HIDDEN_LEN / 8))     # 隐藏层节点数/8
    while(2**n < len_number):
        n += 1
    n = int(math.ceil(n/8))*8
    code += indent + "reg [%d:0] addr;\n"%(n-1)

    # 生成data_use
    code += indent + "wire signed [%d:0] data_use[%d:0];\n"%(
        DATA_WIDTH, DATA_NUMBER-1)
    for i in range(DATA_NUMBER):
        code += indent + "assign data_use[%d] = {1'b0, data[addr][%d:%d];\n"%(
            i, DATA_WIDTH*DATA_NUMBER-1-i*DATA_WIDTH, 
            DATA_WIDTH*DATA_NUMBER-DATA_WIDTH-i*DATA_WIDTH
        )
    debug_signals.append(["data_use", DATA_WIDTH+1, DATA_NUMBER])

    # 生成din_use
    code += indent + "wire signed [%d:0] din_use[%d:0];\n"%(
        DATA_WIDTH, DATA_NUMBER-1)
    for i in range(DATA_NUMBER):
        code += indent + "assign din_use[%d] = {1'b0, din[%d:%d]};\n"%(
            i, DATA_WIDTH*DATA_NUMBER-1-i*DATA_WIDTH, 
            DATA_WIDTH*DATA_NUMBER-DATA_WIDTH-i*DATA_WIDTH
        )
    debug_signals.append(["din_use", DATA_WIDTH+1, DATA_NUMBER])
    
    # 生成待乘数据m1, m2
    code += indent + "reg signed [%d:0] m1[%d:0];\n"%(
        DATA_WIDTH, DATA_NUMBER-1)
    code += indent + "reg signed [%d:0] m2[%d:0];\n"%(
        DATA_WIDTH, DATA_NUMBER-1)
    debug_signals.append(["m1", DATA_WIDTH+1, DATA_NUMBER])
    debug_signals.append(["m2", DATA_WIDTH+1, DATA_NUMBER])

    # 生成mult
    code += indent + "reg signed [%d:0] mult[%d:0];\n"%(
        DATA_WIDTH*2+1, DATA_NUMBER-1)
    debug_signals.append(["mult", DATA_WIDTH*2+2, DATA_NUMBER])

    # 计算需要累加的次数
    accumulate_time = 0
    while(DATA_NUMBER > 2**accumulate_time):
        accumulate_time += 1
    
    # 生成add
    temp_number = DATA_NUMBER   # 当前寄存器数量
    temp_width = DATA_WIDTH*2+2 # 当前寄存器位宽
    for i in range(accumulate_time):
        if(temp_number % 2 == 0):
            temp_number //= 2
        else :
            temp_number = temp_number // 2 + 1
        temp_width += 1
        temp_port = 2**(i+1) if(2**(i+1) < DATA_NUMBER) else DATA_NUMBER
        if(temp_port == DATA_NUMBER):
            code += indent + "reg signed [%d:0] add%d;\n"%(31, temp_port)
            continue
        code += indent + "reg signed [%d:0] add%d[%d:0];\n"%(
            temp_width-1, temp_port, temp_number-1)
        intermediate_reg_number[temp_port] = temp_number
        debug_signals.append(["add%d"%(temp_port), temp_width, temp_number])
    
    # 生成accumulator
    code += indent + "reg signed [31:0] accumulator;\n"

    # 生成last信号
    code += indent + "reg last;\n"

    # 生成valid_pipeline
    code += indent + "reg [%d:0] valid_pipeline;\n"%(accumulate_time+5)
    
    # 生成last_pipeline
    code += indent + "reg [%d:0] last_pipeline;\n"%(accumulate_time+9)

    # 生成channel(用于选择bias)
    bias_len = 0
    for b in BIAS:
        if(len(b) > bias_len):
            bias_len = len(b)
    n = 0
    while(2**n < bias_len):
        n += 1
    n = int(math.ceil(n/8))*8
    code += indent + "reg [%d:0] channel;\n"%(n-1)

    # 生成accu_add_bias
    code += indent + "reg signed [31:0] accu_add_bias;\n"

    # 生成fp_temp
    code += indent + "reg fp_temp_sign;\n"
    code += indent + "reg [47:0] fp_temp;\n"

    # 生成fp_temp_mult_coe
    code += indent + "reg fp_temp_mult_coe_sign;\n"
    code += indent + "reg signed [47:0] fp_temp_mult_coe;\n"

    # 生成t
    code += indent + "reg signed [31:0] t;\n"

    # 生成t_add
    code += indent + "reg signed [31:0] t_add;\n"

    # 生成relu
    code += indent + "reg signed [31:0] relu;\n"

    # 生成bias
    code += indent + "reg [31:0] bias;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(BIAS)):
        indent = "\t\t\t"
        code += indent + "%d'b%s: bias = bias%d[channel];\n"%(
            MUX_WIDTH, decimal_to_binary(i, MUX_WIDTH), i)
    code += indent + "default: bias = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成coe
    code += indent + "reg [47:0] coe;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(COE)):
        indent = "\t\t\t"
        code += indent + "%d'b%s: coe = coe%d;\n"%(
            MUX_WIDTH, decimal_to_binary(i, MUX_WIDTH), i)
    code += indent + "default: coe = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成fp_temp_mult_coe_temp
    code += indent + "wire signed [95:0] fp_temp_mult_coe_temp;\n"
    code += indent + "assign fp_temp_mult_coe_temp = fp_temp * coe;\n"

    # 生成rshfit
    code += indent + "reg [4:0] rshift;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(COE)):
        indent = "\t\t\t"
        code += indent + "%d'b%s: rshift = rshfit%d;\n"%(
            MUX_WIDTH, decimal_to_binary(i, MUX_WIDTH), i)
    code += indent + "default: rshift = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成zero_x
    code += indent + "reg [%d:0] zero_x;\n"%(DATA_WIDTH)
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(COE)):
        indent = "\t\t\t"
        code += indent + "%d'b%s: zero_x = zero_x%d;\n"%(
            MUX_WIDTH, decimal_to_binary(i, MUX_WIDTH), i)
    code += indent + "default: zero_x = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成zero_w
    code += indent + "reg [%d:0] zero_w;\n"%(DATA_WIDTH)
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(COE)):
        indent = "\t\t\t"
        code += indent + "%d'b%s: zero_w = zero_w%d;\n"%(
            MUX_WIDTH, decimal_to_binary(i, MUX_WIDTH), i)
    code += indent + "default: zero_w = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成zero_y
    code += indent + "reg [31:0] zero_y;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(COE)):
        indent = "\t\t\t"
        code += indent + "%d'b%s: zero_y = zero_y%d;\n"%(
            MUX_WIDTH, decimal_to_binary(i, MUX_WIDTH), i)
    code += indent + "default: zero_y = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成last_data_type
    code += indent + "reg last_data_type;\n"

    # 生成计算部分
    code += indent + "always @(posedge clk) begin\n"
    #   # 生成valid, last记录信号
    indent = "\t\t"
    code += indent + "last_data_type <= data_type;\n"
    code += indent + "valid_pipeline <= {valid_pipeline[%d:0], " \
        "in_valid&data_type};\n"%(accumulate_time+4)
    code += indent + "last_pipeline <= {last_pipeline[%d:0], last};\n"%(
        accumulate_time+8)
    #   # 生成data_type=0(feature_map数据)
    code += indent + "if(data_type == 0) begin\n"
    indent = "\t\t\t"
    code += indent + "if(in_valid) begin\n"
    indent = "\t\t\t\t"
    code += indent + "data[addr] <= din;\n"
    code += indent + "if(addr == len-1) begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "addr <= 0;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "addr <= addr + 1;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    indent = "\t\t"
    code += indent + "end\n"
    #   # 生成data_type=1(weight数据)
    code += indent + "else begin\n"
    #   # m=data-zero
    indent = "\t\t\t"
    for i in range(DATA_NUMBER):
        code += indent + "m1[%d] <= data_use[%d] - zero_x;\n"%(i, i)
    for i in range(DATA_NUMBER):
        code += indent + "m2[%d] <= din_use[%d] - zero_w;\n"%(i, i)
    #   # mult=m1*m2
    for i in range(DATA_NUMBER):
        code += indent + "mult[%d] <= m1[%d] * m2[%d];\n"%(i, i, i)
    #   # add2=mult+mult
    temp_data_number = DATA_NUMBER
    if(temp_data_number % 2 == 1):
        temp_data_number = temp_data_number // 2 + 1
        for i in range(temp_data_number - 1):
            code += indent + "add2[%d] <= mult[%d] + mult[%d];\n"%(
                i, i*2, i*2+1)
        code += indent + "add2[%d] <= mult[%d];\n"%(
            temp_data_number-1, (temp_data_number-1)*2)
    else:
        temp_data_number = temp_data_number // 2
        for i in range(temp_data_number):
            code += indent + "add2[%d] <= mult[%d] + mult[%d];\n"%(
                i, i*2, i*2+1)
    #   # add2n=addn+addn
    for i in range(1, accumulate_time):
        if(temp_data_number % 2 == 1):
            if(temp_data_number == 1):
                code += indent + "add%d <= add%d[%d];\n"%(DATA_NUMBER, 2**i, 0)
                continue
            temp_data_number = temp_data_number // 2 + 1
            for j in range(temp_data_number - 1):
                code += indent + "add%d[%d] <= add%d[%d] + add%d[%d];\n"%(
                    2**(i+1), j, 2**i, j*2, 2**i, j*2+1)
            code += indent + "add%d[%d] <= add%d[%d];\n"%(
                2**(i+1), temp_data_number-1, 2**i, (temp_data_number-1)*2)
        else:
            if(temp_data_number == 2):
                code += indent + "add%d <= add%d[%d] + add%d[%d];\n"%(
                    DATA_NUMBER, 2**i, 0, 2**i, 1)
                continue
            temp_data_number = temp_data_number // 2
            for j in range(temp_data_number):
                code += indent + "add%d[%d] <= add%d[%d] + add%d[%d];\n"%(
                    2**(i+1), j, 2**i, j*2, 2**i, j*2+1)
    #   # accumulator+=addn
    #   #   # 重置accumulator
    code += indent + "if(last_data_type == 0 && data_type == 1) begin\n"
    indent = "\t\t\t\t"
    code += indent + "accumulator <= 0;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    #   #   # 累加accumulator
    code += indent + "else begin\n"
    indent = "\t\t\t\t"
    code += indent + "if(valid_pipeline[%d]) begin\n"%(accumulate_time+1)
    indent = "\t\t\t\t\t"
    code += indent + "if(last_pipeline[%d]) begin\n"%(accumulate_time+1)
    indent = "\t\t\t\t\t\t"
    code += indent + "accumulator <= add%d;\n"%(DATA_NUMBER)
    indent = "\t\t\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t\t\t"
    code += indent + "accumulator <= accumulator + add%d;\n"%(DATA_NUMBER)
    indent = "\t\t\t\t\t"
    code += indent + "end\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    #   # temp = accumulator + bias
    code += indent + "if(last_pipeline[%d]) begin\n"%(accumulate_time+1)
    indent = "\t\t\t\t"
    code += indent + "accu_add_bias <= accumulator + bias;\n"
    #   #   # 修改channel
    code += indent + "if(channel == out_len - 1) begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "channel <= 0;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "channel <= channel + 1;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    #   # fp_temp = temp
    code += indent + "fp_temp_sign <= accu_add_bias[31];\n"
    code += indent + "fp_temp <= {{(accu_add_bias - accu_add_bias[31]) ^ " \
        "({32{accu_add_bias[31]}, 16'b0};\n"
    #   # fp_temp *= coe
    code += indent + "fp_temp_mult_coe_sign <= fp_temp_sign;\n"
    code += indent + "fp_temp_mult_coe <= fp_temp_mult_coe_temp[63:16];\n"
    #   # t = fp_temp->to_int >> rshift
    code += indent + "t <= ((fp_temp_mult_coe[47:16] >> rshift) ^ " \
        "{32{fp_temp_mult_coe_sign}}) + fp_temp_mult_coe_sign;\n"
    #   # t = t + zero_y
    code += indent + "t_add <= t + zero_y;\n"
    #   # relu
    code += indent + "if(do_relu) begin\n"
    indent = "\t\t\t\t"
    code += indent + "relu <= (t_add < zero_y) ? zero_y : \n"
    code += indent + "        (t_add > qmax) ? qmax : \n"
    code += indent + "        t_add;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t"
    code += indent + "relu <= (t_add < 0) ? 0 : \n"
    code += indent + "        (t_add > qmax) ? qmax : \n"
    code += indent + "        t_add;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    #  # output
    code += indent + "if(last_pipeline[%d]) begin\n"%(accumulate_time+7)
    indent = "\t\t\t\t"
    code += indent + "dout <= relu[%d:0];\n"%(DATA_WIDTH-1)
    code += indent + "out_valid <= 1;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t"
    code += indent + "out_valid <= 0;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    #   # addr last
    code += indent + "if(in_valid) begin\n"
    indent = "\t\t\t\t"
    code += indent + "if(addr == len-1) begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "last <= 1;\n"
    code += indent + "addr <= 0;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "last <= 0;\n"
    code += indent + "addr <= addr + 1;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t"
    code += indent + "last <= 0;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    indent = "\t\t"
    code += indent + "end\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成initial
    indent = "\t"
    code += indent + "initial begin\n"
    #   # 生成bias
    for n, b in enumerate(BIAS):
        indent = "\t\t"
        for m, b_value in enumerate(b):
            code += indent + "bias%d[%d] = %d;\n"%(n, m, b_value)
    code += indent + "addr = 0;\n"
    code += indent + "channel = 0;\n"
    code += indent + "accumulator = 0;\n"
    for i in range(DATA_NUMBER):
        code += indent + "m1[%d] = 0;\n"%(i)
    for i in range(DATA_NUMBER):
        code += indent + "m2[%d] = 0;\n"%(i)
    for i in range(DATA_NUMBER):
        code += indent + "mult[%d] = 0;\n"%(i)
    temp_number = DATA_NUMBER   # 当前寄存器数量
    for i in range(accumulate_time):
        if(temp_number % 2 == 0):
            temp_number //= 2
        else :
            temp_number = temp_number // 2 + 1
        temp_port = 2**(i+1) if(2**(i+1) < DATA_NUMBER) else DATA_NUMBER
        if(temp_port == DATA_NUMBER):
            code += indent + "add%d = 0;\n"%(temp_port)
            continue
        for j in range(temp_number):
            code += indent + "add%d[%d] = 0;\n"%(temp_port, j)
    indent = "\t"
    code += indent + "end\n"

    # 生成仿真信号
    indent = ""
    code += indent + "// simulation only\n"
    for signal in debug_signals:
        signal_name = signal[0]
        signal_width = signal[1]
        signal_depth = signal[2]
        for i in range(signal_depth):
            code += indent + "wire [%d:0] %s_probe_%d = %s[%d];\n"%(
                signal_width-1, signal_name, i, signal_name, i
            )


    code += "endmodule\n"
    return code