



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


def gen_conv(
    MODULE_NAME: str,   # 模块名
    MUX_WIDTH: int,     # 选择信号位宽
    DATA_WIDTH: int,    # 每个输入数据的位宽，如uint8，则为8
    DATA_NUMBER: int,   # 输入数据数量，如256。这决定了乘累加树的宽度
    OUTPUT_PORTS: list, # 需要的输出端口，如16个累加的结果，256个累加的结果等
    ZERO_X: list,       # ZERO_x数据列表
    ZERO_W: list,       # ZERO_W数据列表
    DEBUG=True,         # 为所有数组信号生成额外的probe信号，方便查看波形
):
    # 需要添加DEBUG信息的信号列表。每个元素是一个列表，
    # 里面有3个值，分别是信号名称，信号位宽，信号数组深度
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
    code += indent + "input [%d:0] mux,\n"%(MUX_WIDTH-1)
    code += indent + "input valid,\n"
    code += indent + "input [%d:0] ina,\n"%(DATA_WIDTH * DATA_NUMBER - 1)
    code += indent + "input [%d:0] inb,\n"%(DATA_WIDTH * DATA_NUMBER - 1)
    for port in OUTPUT_PORTS:
        # 需要计算当前中间结果的个数，以及每个的位宽
        # 输入的数据位宽为DATA_WIDTH，无符号扩展后+1，乘法后*2，之后每累加一次+1。
        # 如果当前输出端口累加次数>2^n，且<=2^(n+1)，则需要累加n+1次
        # 另外，如果输出最终累加结果，直接输出32bit
        if(port == DATA_NUMBER):
            code += indent + "output [31:0] add%d_o,\n"%(port)
            code += indent + "output valid%d,\n"%(port)
            continue
        base = (DATA_WIDTH + 1) * 2
        n = 0
        while(port > 2**n):
            n += 1
        width_per_data = base + n
        number = DATA_NUMBER // port
        width = width_per_data * number
        code += indent + "output [%d:0] add%d_o,\n"%(width-1, port)
        code += indent + "output valid%d,\n"%(port)
    code = code[:-2] + "\n" # 去除最后一个逗号
    code += ");\n"

    # 计算需要累加的次数
    accumulate_time = 0
    while(DATA_NUMBER > 2**accumulate_time):
        accumulate_time += 1
    
    # 生成valid_pipeline
    indent = "\t"
    code += indent + "reg [%d:0] valid_pipeline;\n"%(accumulate_time + 5)

    # 生成ina_use, inb_use
    code += indent + "wire signed [%d:0] ina_use[%d:0];\n"%(
        DATA_WIDTH, DATA_NUMBER-1)
    for i in range(DATA_NUMBER):
        code += indent + "assign ina_use[%d] = {1'b0, ina[%d:%d]};\n"%(
            i, DATA_WIDTH*i+DATA_WIDTH-1, DATA_WIDTH*i)
    debug_signals.append(["ina_use", DATA_WIDTH+1, DATA_NUMBER])
    code += indent + "wire signed [%d:0] inb_use[%d:0];\n"%(
        DATA_WIDTH, DATA_NUMBER-1)
    for i in range(DATA_NUMBER):
        code += indent + "assign inb_use[%d] = {1'b0, inb[%d:%d]};\n"%(
            i, DATA_WIDTH*i+DATA_WIDTH-1, DATA_WIDTH*i)
    debug_signals.append(["inb_use", DATA_WIDTH+1, DATA_NUMBER])

    # 生成data_a, data_b
    code += indent + "reg signed [%d:0] data_a[%d:0];\n"%(
        DATA_WIDTH, DATA_NUMBER-1)
    code += indent + "reg signed [%d:0] data_b[%d:0];\n"%(
        DATA_WIDTH, DATA_NUMBER-1)
    debug_signals.append(["data_a", DATA_WIDTH+1, DATA_NUMBER])
    debug_signals.append(["data_b", DATA_WIDTH+1, DATA_NUMBER])

    # 生成mult
    code += indent + "reg signed [%d:0] mult[%d:0];\n"%(
        DATA_WIDTH*2+1, DATA_NUMBER-1)
    debug_signals.append(["mult", DATA_WIDTH*2, DATA_NUMBER])

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

    # 为valid信号赋值
    for port in OUTPUT_PORTS:
        n = 0
        while(port > 2**n):
            n += 1
        code += indent + "assign valid%d = valid_pipeline[%d];\n"%(port, n+1)
    
    # 为输出信号赋值
    for port in OUTPUT_PORTS:
        if(port == DATA_NUMBER):
            code += indent + "assign add%d_o = add%d;\n"%(port, port)
            continue
        code += indent + "assign add%d_o = {"%(port)
        output_number = DATA_NUMBER // port     # 实际能够输出的数量
        for i in range(output_number):
            code += "add%d[%d], "%(port, intermediate_reg_number[port]-i-1)
        code = code[:-2] + "};\n"
    
    # 生成zero_x parameter
    for n,x in enumerate(ZERO_X):
        code += indent + "parameter signed [%d:0] zero_x%d = %d;\n"%(
            DATA_WIDTH, n, x)
    
    # 生成zero_w parameter
    for n,w in enumerate(ZERO_W):
        code += indent + "parameter signed [%d:0] zero_w%d = %d;\n"%(
            DATA_WIDTH, n, w)
    
    # 生成zero_x
    code += indent + "reg signed [%d:0] zero_x;\n"%(DATA_WIDTH)
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for n,x in enumerate(ZERO_X):
        indent = "\t\t\t"
        code += indent + "%d'b%s: zero_x = zero_x%d;\n"%(
            MUX_WIDTH, decimal_to_binary(n, MUX_WIDTH), n)
    code += indent + "default: zero_x = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成zero_w
    code += indent + "reg signed [%d:0] zero_w;\n"%(DATA_WIDTH)
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for n,w in enumerate(ZERO_W):
        indent = "\t\t\t"
        code += indent + "%d'b%s: zero_w = zero_w%d;\n"%(
            MUX_WIDTH, decimal_to_binary(n, MUX_WIDTH), n)
    code += indent + "default: zero_w = 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # 生成流水计算部分
    code += indent + "always @(posedge clk) begin\n"
    indent = "\t\t"
    #   # 生成valid_pipeline
    code += indent + "valid_pipeline <= {valid_pipeline[%d:0], valid};\n"%(
        accumulate_time+4)
    #   # in -> data
    for i in range(DATA_NUMBER):
        code += indent + "data_a[%d] <= ina_use[%d] - zero_w;\n"%(i, i)
    for i in range(DATA_NUMBER):
        code += indent + "data_b[%d] <= inb_use[%d] - zero_x;\n"%(i, i)
    #   # data -> mult
    for i in range(DATA_NUMBER):
        code += indent + "mult[%d] <= data_a[%d] * data_b[%d];\n"%(i, i, i)
    #   # mult -> add2
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
    #   # addn -> add2n
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
    indent = "\t"
    code += indent + "end\n"

    # 生成initial
    code += indent + "initial begin\n"
    indent = "\t\t"
    code += indent + "valid_pipeline <= 0;\n"
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