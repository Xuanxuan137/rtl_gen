

def decimal_to_bin(value, width):
    '''
    transfer decimal value to binary with 'width' width
    '''
    if(value < 0):
        raise ValueError("value should be non-negative")
    code = ""
    for i in range(width):
        if(value >= 2**(width-1-i)):
            code += "1"
            value -= 2**(width-1-i)
        else:
            code += "0"
    return code


def coe_to_bin(data):
    '''
    convert a coe value to binary
    '''
    bin = ""
    temp = 0.0
    for i in range(16):
        if(temp + 2**(-1-i) > data):
            bin += "0"
        else:
            bin += "1"
            temp += 2**(-1-i)
    return bin


def gen_add(
    MODULE_NAME: str,           # module name
    MUX_WIDTH: int,             # mux width
    TOTAL_COUNT_WIDTH: int,     # total count width(total_count:
                                # amount of data to add)
    COE: tuple,                 # (coe1, coe2)
    RSHIFT: tuple,              # (rshift1, rshift2)
    ZERO_X: tuple,              # (zero_x1, zero_x2)
    ZERO_Y: list,               # zero_y
    QMAX: int,                  # qmax
    QMIN: int,                  # qmin
    DEUG=True                   # debug
):
    '''
    Generate add module
    '''

    # each item in debug_signals: [signal_name, signal_width, signal_depth]
    debug_signals = []

    # the code to return
    code = ""

    # generate module
    code += "module %s(\n"%(MODULE_NAME)

    # generate ports
    dma_width = 64          # dma tdata width
    dma_data_width = 8      # 8 bits per data
    dma_count = dma_width // dma_data_width     # number of data in dma_tdata
    indent = "\t"
    code += indent + "input clk,\n"
    code += indent + "input reset,\n"
    code += indent + "input [%d:0] total_count,\n"%(TOTAL_COUNT_WIDTH-1)
    code += indent + "input [%d:0] r0_tdata,\n"%(dma_width-1)
    code += indent + "input r0_tvalid,\n"
    code += indent + "input r0_tready,\n"
    code += indent + "input [%d:0] r1_tdata,\n"%(dma_width-1)
    code += indent + "input r1_tvalid,\n"
    code += indent + "input r1_tready,\n"
    code += indent + "input w0_tready,\n"
    code += indent + "input [%d:0] mux,\n"%(MUX_WIDTH-1)
    code += indent + "output add_r0_tready,\n"
    code += indent + "output add_r1_tready,\n"
    code += indent + "output [%d:0] w0_tdata,\n"%(dma_width-1)
    code += indent + "output w0_tvalid,\n"
    code += indent + "output w0_tlast\n"
    code += ");\n"

    # generate fifo
    force_register = True
    code += indent + ("(* ram_style =  \"registers\" *)" if(force_register)
        else "") + "reg [%d:0] fifo0[15:0];\n"%(dma_width-1)
    code += indent + ("(* ram_style =  \"registers\" *)" if(force_register)
        else "") + "reg [%d:0] fifo1[15:0];\n"%(dma_width-1)

    # generate fifo addr
    code += indent + "reg [3:0] raddr0;\n"
    code += indent + "reg [3:0] waddr0;\n"
    code += indent + "reg [3:0] raddr1;\n"
    code += indent + "reg [3:0] waddr1;\n"

    # generate counter
    code += indent + "reg [%d:0] count;\n"%(TOTAL_COUNT_WIDTH-1)

    # generate fifo address controller
    code += indent + "assign add_r0_tready = ((raddr0 - waddr0 < 4'h5) && " \
         "(raddr0 != waddr0)) ? 0 : 1;\n"
    code += indent + "assign add_r1_tready = ((raddr1 - waddr1 < 4'h5) && " \
         "(raddr1 != waddr1)) ? 0 : 1;\n"

    # generate fifo empty signal
    code += indent + "wire empty0;\n"
    code += indent + "wire empty1;\n"
    code += indent + "assign empty0 = (raddr0 == waddr0) ? 1 : 0;\n"
    code += indent + "assign empty1 = (raddr1 == waddr1) ? 1 : 0;\n"

    # generate coe
    coe1 = COE[0]
    coe2 = COE[1]
    for n, c in enumerate(coe1):
        code += indent + "parameter [47:0] coe1_%d = 48'b%s;\n"%(n,
            coe_to_bin(c))
    for n, c in enumerate(coe2):
        code += indent + "parameter [47:0] coe2_%d = 48'b%s;\n"%(n,
            coe_to_bin(c))

    # generate rshift(for direction, 0 for right shift, 1 for left)
    rshift1 = RSHIFT[0]
    rshift2 = RSHIFT[1]
    for n, r in enumerate(rshift1):
        sign = 1 if(r < 0) else 0
        r = abs(r)
        code += indent + "parameter [4:0] rshift1_%d = %d;\n"%(n, r)
        code += indent + "parameter shift_direction1_%d = %d;\n"%(n, sign)
    for n, r in enumerate(rshift2):
        sign = 1 if(r < 0) else 0
        r = abs(r)
        code += indent + "parameter [4:0] rshift2_%d = %d;\n"%(n, r)
        code += indent + "parameter shift_direction2_%d = %d;\n"%(n, sign)

    # generate zero_x
    zero_x1 = ZERO_X[0]
    zero_x2 = ZERO_X[1]
    for n, x in enumerate(zero_x1):
        code += indent + "parameter signed [8:0] zero_x1_%d = %d;\n"%(n, x)
    for n, x in enumerate(zero_x2):
        code += indent + "parameter signed [8:0] zero_x2_%d = %d;\n"%(n, x)

    # generate zero_y
    for n, y in enumerate(ZERO_Y):
        code += indent + "parameter signed [31:0] zero_y_%d = %d;\n"%(n, y)

    # generate qmin qmax
    code += indent + "parameter [31:0] qmin = %d;\n"%(QMIN)
    code += indent + "parameter [31:0] qmax = %d;\n"%(QMAX)

    # generate adder unsigned extend
    code += indent + "reg signed [8:0] adder1_use[%d:0];\n"%(dma_count-1)
    code += indent + "reg signed [8:0] adder2_use[%d:0];\n"%(dma_count-1)
    code += indent + "reg adder_use_valid;\n"

    # generate adder
    code += indent + "reg signed [8:0] adder1[%d:0];\n"%(dma_count-1)
    code += indent + "reg signed [8:0] adder2[%d:0];\n"%(dma_count-1)
    code += indent + "reg adder_valid;\n"

    # generate fp_temp
    code += indent + "reg fp_temp1_sign[%d:0];\n"%(dma_count-1)
    code += indent + "reg [47:0] fp_temp1[%d:0];\n"%(dma_count-1)
    code += indent + "reg fp_temp2_sign[%d:0];\n"%(dma_count-1)
    code += indent + "reg [47:0] fp_temp2[%d:0];\n"%(dma_count-1)
    code += indent + "reg fp_temp_valid;\n"

    # generate fp_temp_mult_coe
    code += indent + "reg fp_temp1_mult_coe1_sign[%d:0];\n"%(dma_count-1)
    code += indent + "reg signed [47:0] fp_temp1_mult_coe1[%d:0];\n"%(
        dma_count-1)
    code += indent + "reg fp_temp2_mult_coe2_sign[%d:0];\n"%(dma_count-1)
    code += indent + "reg signed [47:0] fp_temp2_mult_coe2[%d:0];\n"%(
        dma_count-1)
    code += indent + "reg fp_temp_mult_coe_valid;\n"

    # generate t
    code += indent + "reg signed [31:0] t1[%d:0];\n"%(dma_count-1)
    code += indent + "reg signed [31:0] t2[%d:0];\n"%(dma_count-1)
    code += indent + "reg t_valid;\n"

    # generate t_shift
    code += indent + "reg signed [31:0] t1_shift[%d:0];\n"%(dma_count-1)
    code += indent + "reg signed [31:0] t2_shift[%d:0];\n"%(dma_count-1)
    code += indent + "reg t_shift_valid;\n"

    # generate t_add
    code += indent + "reg signed [31:0] t_add[%d:0];\n"%(dma_count-1)
    code += indent + "reg t_add_valid;\n"

    # genereate t_add_y
    code += indent + "reg signed [31:0] t_add_y[%d:0];\n"%(dma_count-1)
    code += indent + "reg t_add_y_valid;\n"

    # generate result
    code += indent + "reg [7:0] result[%d:0];\n"%(dma_count-1)
    code += indent + "reg result_valid;\n"
    code += indent + "reg last_result;\n"

    # generate coe
    code += indent + "reg [47:0] coe1;\n"
    code += indent + "reg [47:0] coe2;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(COE[0])):
        indent = "\t\t\t"
        code += indent + "%d'b%s: begin\n"%(MUX_WIDTH, decimal_to_bin(i,
            MUX_WIDTH))
        indent = "\t\t\t\t"
        code += indent + "coe1 <= coe1_%d;\n"%(i)
        code += indent + "coe2 <= coe2_%d;\n"%(i)
        indent = "\t\t\t"
        code += indent + "end\n"
    code += indent + "default: begin\n"
    indent = "\t\t\t\t"
    code += indent + "coe1 <= 0;\n"
    code += indent + "coe2 <= 0;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # generate fp_temp_mult_coe_temp
    code += indent + "wire signed [95:0] fp_temp1_mult_coe1_temp[%d:0];\n"%(
        dma_count-1)
    for i in range(dma_count):
        code += indent + "assign fp_temp1_mult_coe1_temp[%d] = fp_temp1[%d]" \
             " * coe1;\n"%(i, i)
    code += indent + "wire signed [95:0] fp_temp2_mult_coe2_temp[%d:0];\n"%(
        dma_count-1)
    for i in range(dma_count):
        code += indent + "assign fp_temp2_mult_coe2_temp[%d] = fp_temp2[%d]" \
             " * coe2;\n"%(i, i)

    # generate rshift
    code += indent + "reg [4:0] rshift1;\n"
    code += indent + "reg shift_direction1;\n"
    code += indent + "reg [4:0] rshift2;\n"
    code += indent + "reg shift_direction2;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(RSHIFT[0])):
        indent = "\t\t\t"
        code += indent + "%d'b%s: begin\n"%(MUX_WIDTH, decimal_to_bin(
            i, MUX_WIDTH))
        indent = "\t\t\t\t"
        code += indent + "rshift1 <= rshift1_%d;\n"%(i)
        code += indent + "shift_direction1 <= shift_direction1_%d;\n"%(i)
        code += indent + "rshift2 <= rshift2_%d;\n"%(i)
        code += indent + "shift_direction2 <= shift_direction2_%d;\n"%(i)
        indent = "\t\t\t"
        code += indent + "end\n"
    code += indent + "default: begin\n"
    indent = "\t\t\t\t"
    code += indent + "rshift1 <= 0;\n"
    code += indent + "shift_direction1 <= 0;\n"
    code += indent + "rshift2 <= 0;\n"
    code += indent + "shift_direction2 <= 0;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # generate zero_x
    code += indent + "reg signed [8:0] zero_x1;\n"
    code += indent + "reg signed [8:0] zero_x2;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(ZERO_X[0])):
        indent = "\t\t\t"
        code += indent + "%d'b%s: begin\n"%(MUX_WIDTH, decimal_to_bin(
            i, MUX_WIDTH))
        indent = "\t\t\t\t"
        code += indent + "zero_x1 <= zero_x1_%d;\n"%(i)
        code += indent + "zero_x2 <= zero_x2_%d;\n"%(i)
        indent = "\t\t\t"
        code += indent + "end\n"
    code += indent + "default: begin\n"
    indent = "\t\t\t\t"
    code += indent + "zero_x1 <= 0;\n"
    code += indent + "zero_x2 <= 0;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # generate zero_y
    code += indent + "reg signed [31:0] zero_y;\n"
    code += indent + "always @(*) begin\n"
    indent = "\t\t"
    code += indent + "case(mux)\n"
    for i in range(len(ZERO_Y)):
        indent = "\t\t\t"
        code += indent + "%d'b%s: zero_y <= zero_y_%d;\n"%(MUX_WIDTH,
            decimal_to_bin(i, MUX_WIDTH), i)
    code += indent + "default: zero_y <= 0;\n"
    indent = "\t\t"
    code += indent + "endcase\n"
    indent = "\t"
    code += indent + "end\n"

    # assign value to dma_w0
    code += indent + "assign w0_tdata = {\n"
    indent = "\t\t"
    for i in range(dma_count):
        code += indent + "result[%d],\n"%(dma_count-1-i)
    code = code[:-2] + "\n"
    indent = "\t"
    code += indent + "};\n"
    code += indent + "assign w0_tvalid = result_valid;\n"
    code += indent + "assign w0_tlast = last_result;\n"

    # calculate
    code += indent + "always @(posedge clk) begin\n"
    indent = "\t\t"

    # # reset
    code += indent + "if(reset) begin\n"
    indent = "\t\t\t"
    code += indent + "raddr0 <= 0;\n"
    code += indent + "raddr1 <= 0;\n"
    code += indent + "count <= 0;\n"
    code += indent + "adder_use_valid <= 0;\n"
    code += indent + "adder_valid <= 0;\n"
    code += indent + "fp_temp_valid <= 0;\n"
    code += indent + "fp_temp_mult_coe_valid <= 0;\n"
    code += indent + "t_valid <= 0;\n"
    code += indent + "t_shift_valid <= 0;\n"
    code += indent + "t_add_valid <= 0;\n"
    code += indent + "t_add_y_valid <= 0;\n"
    code += indent + "result_valid <= 0;\n"
    code += indent + "last_result <= 0;\n"
    indent = "\t\t"
    code += indent + "end\n"

    # # calculate
    code += indent + "else begin\n"
    indent = "\t\t\t"
    code += indent + "if(w0_tready) begin\n"
    indent = "\t\t\t\t"
    # # # read data from fifo
    code += indent + "if(!empty0 & !empty1) begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "adder_use_valid <= 1;\n"
    for i in range(dma_count):
        code += indent + "adder1_use[%d] <= {1'b0, fifo0[raddr0][%d:%d]};\n"%(
            i, i*8+7, i*8)
    for i in range(dma_count):
        code += indent + "adder2_use[%d] <= {1'b0, fifo1[raddr1][%d:%d]};\n"%(
            i, i*8+7, i*8)
    code += indent + "raddr0 <= raddr0 + 1;\n"
    code += indent + "raddr1 <= raddr1 + 1;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "adder_use_valid <= 0;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    # # # temp = temp_x - zero_x
    for i in range(dma_count):
        code += indent + "adder1[%d] <= adder1_use[%d] - zero_x1;\n"%(i, i)
    for i in range(dma_count):
        code += indent + "adder2[%d] <= adder2_use[%d] - zero_x2;\n"%(i, i)
    code += indent + "adder_valid <= adder_use_valid;\n"
    # # # fp_temp.assign(temp)
    for i in range(dma_count):
        code += indent + "fp_temp1_sign[%d] <= adder1[%d][8];\n"%(i, i)
        code += indent + "fp_temp1[%d] <= {23'b0, {(adder1[%d] - adder1[%d]" \
            "[8]) ^ ({9{adder1[%d][8]}})}, 16'b0};\n"%(i, i, i, i)
    for i in range(dma_count):
        code += indent + "fp_temp2_sign[%d] <= adder2[%d][8];\n"%(i, i)
        code += indent + "fp_temp2[%d] <= {23'b0, {(adder2[%d] - adder2[%d]" \
            "[8]) ^ ({9{adder2[%d][8]}})}, 16'b0};\n"%(i, i, i, i)
    code += indent + "fp_temp_valid <= adder_valid;\n"
    # # # fp_temp *= coe
    for i in range(dma_count):
        code += indent + "fp_temp1_mult_coe1_sign[%d] <= fp_temp1_sign[%d]" \
            ";\n"%(i, i)
        code += indent + "fp_temp1_mult_coe1[%d] <= fp_temp1_mult_coe1_temp" \
            "[%d][63:16];\n"%(i, i)
    for i in range(dma_count):
        code += indent + "fp_temp2_mult_coe2_sign[%d] <= fp_temp2_sign[%d]" \
            ";\n"%(i, i)
        code += indent + "fp_temp2_mult_coe2[%d] <= fp_temp2_mult_coe2_temp" \
            "[%d][63:16];\n"%(i, i)
    code += indent + "fp_temp_mult_coe_valid <= fp_temp_valid;\n"
    # # # t = fp_temp.to_int()
    for i in range(dma_count):
        code += indent + "t1[%d] <= (fp_temp1_mult_coe1[%d][47:16] ^ {32{" \
            "fp_temp1_mult_coe1_sign[%d]}}) + fp_temp1_mult_coe1_sign[%d];\n"%(
            i, i, i, i)
    for i in range(dma_count):
        code += indent + "t2[%d] <= (fp_temp2_mult_coe2[%d][47:16] ^ {32{" \
            "fp_temp2_mult_coe2_sign[%d]}}) + fp_temp2_mult_coe2_sign[%d];\n"%(
            i, i, i, i)
    code += indent + "t_valid <= fp_temp_mult_coe_valid;\n"
    # # # shift
    code += indent + "if(shift_direction1) begin\n"
    indent = "\t\t\t\t\t"
    for i in range(dma_count):
        code += indent + "t1_shift[%d] <= t1[%d] << rshift1;\n"%(i, i)
    indent = "\t\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t\t"
    for i in range(dma_count):
        code += indent + "t1_shift[%d] <= t1[%d] >>> rshift1;\n"%(i, i)
    indent = "\t\t\t\t"
    code += indent + "end\n"
    code += indent + "if(shift_direction2) begin\n"
    indent = "\t\t\t\t\t"
    for i in range(dma_count):
        code += indent + "t2_shift[%d] <= t2[%d] << rshift2;\n"%(i, i)
    indent = "\t\t\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t\t\t"
    for i in range(dma_count):
        code += indent + "t2_shift[%d] <= t2[%d] >>> rshift2;\n"%(i, i)
    indent = "\t\t\t\t"
    code += indent + "end\n"
    code += indent + "t_shift_valid <= t_valid;\n"
    # # # add
    for i in range(dma_count):
        code += indent + "t_add[%d] <= t1_shift[%d] + t2_shift[%d];\n"%(
            i, i, i)
    code += indent + "t_add_valid <= t_shift_valid;\n"
    # # # add zero_y
    for i in range(dma_count):
        code += indent + "t_add_y[%d] <= t_add[%d] + zero_y;\n"%(i, i)
    code += indent + "t_add_y_valid <= t_add_valid;\n"
    # # # clip
    for i in range(dma_count):
        code += indent + "result[%d] <= (t_add_y[%d] < qmin) ? qmin : \n"%(i,i)
        code += indent + "             (t_add_y[%d] > qmax) ? qmax : \n"%(i)
        code += indent + "             t_add_y[%d];\n"%(i)
    code += indent + "result_valid <= t_add_y_valid;\n"
    # # # update count
    code += indent + "if(result_valid) begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "count <= count + 1;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    # # # update last_result
    code += indent + "if(count == total_count-2) begin\n"
    indent = "\t\t\t\t\t"
    code += indent + "last_result <= 1;\n"
    indent = "\t\t\t\t"
    code += indent + "end\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    indent = "\t\t"
    code += indent + "end\n"
    indent = "\t"
    code += indent + "end\n"


    # write fifo
    code += indent + "always @(posedge clk) begin\n"
    indent = "\t\t"
    code += indent + "if(reset) begin\n"
    indent = "\t\t\t"
    code += indent + "waddr0 <= 0;\n"
    code += indent + "waddr1 <= 0;\n"
    indent = "\t\t"
    code += indent + "end\n"
    code += indent + "else begin\n"
    indent = "\t\t\t"
    code += indent + "if(r0_tready) begin\n"
    indent = "\t\t\t\t"
    code += indent + "fifo0[waddr0] <= r0_tdata;\n"
    code += indent + "waddr0 <= waddr0 + 1;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    code += indent + "if(r1_tready) begin\n"
    indent = "\t\t\t\t"
    code += indent + "fifo1[waddr1] <= r1_tdata;\n"
    code += indent + "waddr1 <= waddr1 + 1;\n"
    indent = "\t\t\t"
    code += indent + "end\n"
    indent = "\t\t"
    code += indent + "end\n"
    indent = "\t"
    code += indent + "end\n"


    # initial
    code += indent + "initial begin\n"
    indent = "\t\t"
    code += indent + "raddr0 = 0;\n"
    code += indent + "waddr0 = 0;\n"
    code += indent + "raddr1 = 0;\n"
    code += indent + "waddr1 = 0;\n"
    code += indent + "count = 0;\n"
    for i in range(dma_count):
        code += indent + "adder1_use[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "adder2_use[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "adder1[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "adder2[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "fp_temp1_sign[%d] = 0;\n"%(i)
        code += indent + "fp_temp1[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "fp_temp2_sign[%d] = 0;\n"%(i)
        code += indent + "fp_temp2[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "fp_temp1_mult_coe1_sign[%d] = 0;\n"%(i)
        code += indent + "fp_temp1_mult_coe1[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "fp_temp2_mult_coe2_sign[%d] = 0;\n"%(i)
        code += indent + "fp_temp2_mult_coe2[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "t1[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "t2[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "t1_shift[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "t2_shift[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "t_add[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "t_add_y[%d] = 0;\n"%(i)
    for i in range(dma_count):
        code += indent + "result[%d] = 0;\n"%(i)
    code += indent + "adder_use_valid = 0;\n"
    code += indent + "adder_valid = 0;\n"
    code += indent + "fp_temp_valid = 0;\n"
    code += indent + "fp_temp_mult_coe_valid = 0;\n"
    code += indent + "t_valid = 0;\n"
    code += indent + "t_shift_valid = 0;\n"
    code += indent + "t_add_valid = 0;\n"
    code += indent + "t_add_y_valid = 0;\n"
    code += indent + "result_valid = 0;\n"
    code += indent + "last_result = 0;\n"
    indent = "\t"
    code += indent + "end\n"


    code += "endmodule"



    return code