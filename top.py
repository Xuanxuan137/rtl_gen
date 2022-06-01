


import math


def gen_top(
    WEIGHT_BUF_COL: int,        # number of weight buffer columns
    WEIGUT_BUF_DEPTH: int,      # depth of weight buffer
    FEATURE_MAP_BUF_COL: int,   # number of feature map buffer columns
    FEATURE_MAP_BUF_DEPTH: int, # depth of feature map buffer
    OUTPUT_BUF_COL: int,        # number of output buffer columns
    OUTPUT_BUF_DEPTH: int,      # depth of output buffer
):
    '''
    Gen top module of the accelerator
    '''
    # Signals need to debug. Each element is a list, which has 3 values, 
    # which are signal name, signal width, and signal array depth
    debug_signals = []

    code = ""

    # module name
    code += "module main(\n"

    # input output ports
    indent = "\t"

    code += ");\n"

    # global signals
    code += indent + "wire clk;\n"

    # signals for block design
    code += indent + "reg [31:0] bram_addr;\n"
    code += indent + "reg [31:0] bram_din;\n"
    code += indent + "wire [31:0] bram_dout;\n"
    code += indent + "reg bram_en;\n"
    code += indent + "reg bram_rst;\n"
    code += indent + "reg [3:0] bram_we;\n"
    code += indent + "wire [63:0] dma_r0_tdata;\n"
    code += indent + "wire [7:0] dma_r0_tkeep;\n"
    code += indent + "wire dma_r0_tlast;\n"
    code += indent + "reg dma_r0_tready;\n"
    code += indent + "wire dma_r0_tvalid;\n"
    code += indent + "wire [63:0] dma_r1_tdata;\n"
    code += indent + "wire [7:0] dma_r1_tkeep;\n"
    code += indent + "wire dma_r1_tlast;\n"
    code += indent + "reg dma_r1_tready;\n"
    code += indent + "wire dma_r1_tvalid;\n"
    code += indent + "reg [63:0] dma_w0_tdata;\n"
    code += indent + "reg [7:0] dma_w0_tkeep;\n"
    code += indent + "reg dma_w0_tlast;\n"
    code += indent + "wire dma_w0_tready;\n"
    code += indent + "reg dma_w0_tvalid;\n"

    # block design
    code += indent + "design_1_wrapper d1w1 (\n"
    indent = "\t\t"
    code += indent + ".BRAM_PORTB_0_addr(bram_addr),\n"
    code += indent + ".BRAM_PORTB_0_clk(clk),\n"
    code += indent + ".BRAM_PORTB_0_din(bram_din),\n"
    code += indent + ".BRAM_PORTB_0_dout(bram_dout),\n"
    code += indent + ".BRAM_PORTB_0_en(bram_en),\n"
    code += indent + ".BRAM_PORTB_0_rst(bram_rst),\n"
    code += indent + ".BRAM_PORTB_0_we(bram_we),\n"
    code += indent + ".FCLK_CLK0_0(clk),\n"
    code += indent + ".M_AXIS_0_tdata(dma_r0_tdata),\n"
    code += indent + ".M_AXIS_0_tkeep(dma_r0_tkeep),\n"
    code += indent + ".M_AXIS_0_tlast(dma_r0_tlast),\n"
    code += indent + ".M_AXIS_0_tready(dma_r0_tready),\n"
    code += indent + ".M_AXIS_0_tvalid(dma_r0_tvalid),\n"
    code += indent + ".M_AXIS_1_tdata(dma_r1_tdata),\n"
    code += indent + ".M_AXIS_1_tkeep(dma_r1_tkeep),\n"
    code += indent + ".M_AXIS_1_tlast(dma_r1_tlast),\n"
    code += indent + ".M_AXIS_1_tready(dma_r1_tready),\n"
    code += indent + ".M_AXIS_1_tvalid(dma_r1_tvalid),\n"
    code += indent + ".S_AXIS_0_tdata(dma_w0_tdata),\n"
    code += indent + ".S_AXIS_0_tkeep(dma_w0_tkeep),\n"
    code += indent + ".S_AXIS_0_tlast(dma_w0_tlast),\n"
    code += indent + ".S_AXIS_0_tready(dma_w0_tready),\n"
    code += indent + ".S_AXIS_0_tvalid(dma_w0_tvalid)\n"
    indent = "\t"
    code += indent + ");\n"

    # buffer(weight buffer, feature map buffer, output buffer)
    # bram_A port
    weight_buf_addr_bit_width = math.ceil(math.log2(WEIGUT_BUF_DEPTH))
    feature_map_buf_addr_bit_width = math.ceil(math.log2(
        FEATURE_MAP_BUF_DEPTH))
    code += indent + "reg bram_a_ena[%d:0];\n"%(WEIGHT_BUF_COL-1)
    code += indent + "reg bram_a_wea[%d:0];\n"%(WEIGHT_BUF_COL-1)
    code += indent + "reg [%d:0] bram_a_addra[%d:0];\n"%(
        weight_buf_addr_bit_width-1, WEIGHT_BUF_COL-1)
    code += indent + "reg [63:0] bram_a_dina[%d:0];\n"%(WEIGHT_BUF_COL-1)
    code += indent + "reg bram_a_enb[%d:0];\n"%(WEIGHT_BUF_COL-1)
    code += indent + "reg [%d:0] bram_a_addrb[%d:0];\n"%(
        weight_buf_addr_bit_width-1, WEIGHT_BUF_COL-1)
    code += indent + "wire [63:0] bram_a_doutb[%d:0];\n"%(WEIGHT_BUF_COL-1)
    # bram_B port
    code += indent + "reg bram_b_ena[%d:0];\n"%(FEATURE_MAP_BUF_COL-1)
    code += indent + "reg bram_b_wea[%d:0];\n"%(WEIGHT_BUF_COL-1)
    code += indent + "reg [%d:0] bram_b_addra[%d:0];\n"%(
        feature_map_buf_addr_bit_width-1, FEATURE_MAP_BUF_COL-1)
    code += indent + "reg [63:0] bram_b_dina[%d:0];\n"%(FEATURE_MAP_BUF_COL-1)
    code += indent + "reg bram_b_enb[%d:0];\n"%(FEATURE_MAP_BUF_COL-1)
    code += indent + "reg [%d:0] bram_b_addrb[%d:0];\n"%(
        feature_map_buf_addr_bit_width-1, FEATURE_MAP_BUF_COL-1)
    code += indent + "wire [63:0] bram_b_doutb[%d:0];\n"%(FEATURE_MAP_BUF_COL-1)
    # instantiate bram_A
    for i in range(WEIGHT_BUF_COL):
        code += indent + "blk_64_512 bram_a_%d (\n"%(i)
        indent = "\t\t"
        code += indent + ".clka(clk),\n"
        code += indent + ".ena(bram_a_ena[%d]),\n"%(i)
        code += indent + ".wea(bram_a_wea[%d]),\n"%(i)
        code += indent + ".addra(bram_a_addra[%d]),\n"%(i)
        code += indent + ".dina(bram_a_dina[%d]),\n"%(i)
        code += indent + ".clkb(clk),\n"
        code += indent + ".enb(bram_a_enb[%d]),\n"%(i)
        code += indent + ".addrb(bram_a_addrb[%d]),\n"%(i)
        code += indent + ".doutb(bram_a_doutb[%d])\n"%(i)
        indent = "\t"
        code += indent + ");\n"
    # instantiate bram_B
    for i in range(FEATURE_MAP_BUF_COL):
        code += indent + "blk_64_512 bram_b_%d (\n"%(i)
        indent = "\t\t"
        code += indent + ".clka(clk),\n"
        code += indent + ".ena(bram_b_ena[%d]),\n"%(i)
        code += indent + ".wea(bram_b_wea[%d]),\n"%(i)
        code += indent + ".addra(bram_b_addra[%d]),\n"%(i)
        code += indent + ".dina(bram_b_dina[%d]),\n"%(i)
        code += indent + ".clkb(clk),\n"
        code += indent + ".enb(bram_b_enb[%d]),\n"%(i)
        code += indent + ".addrb(bram_b_addrb[%d]),\n"%(i)
        code += indent + ".doutb(bram_b_doutb[%d])\n"%(i)
        indent = "\t"
        code += indent + ");\n"
    # bram_C port
    output_buf_addr_bit_width = math.ceil(math.log2(OUTPUT_BUF_DEPTH))
    code += indent + "reg bram_r_ena[%d:0];\n"%(OUTPUT_BUF_COL-1)
    code += indent + "reg bram_r_wea[%d:0];\n"%(OUTPUT_BUF_COL-1)
    code += indent + "reg [%d:0] bram_r_addra[%d:0];\n"%(
        output_buf_addr_bit_width-1, OUTPUT_BUF_COL-1)
    code += indent + "reg [63:0] bram_r_dina[%d:0];\n"%(OUTPUT_BUF_COL-1)
    code += indent + "reg bram_r_enb[%d:0];\n"%(OUTPUT_BUF_COL-1)
    code += indent + "reg [%d:0] bram_r_addrb[%d:0];\n"%(
        output_buf_addr_bit_width-1, OUTPUT_BUF_COL-1)
    code += indent + "wire [63:0] bram_r_doutb[%d:0];\n"%(OUTPUT_BUF_COL-1)
    # instantiate bram_C
    for i in range(OUTPUT_BUF_COL):
        code += indent + "blk_64_%d bram_r_%d (\n"%(OUTPUT_BUF_DEPTH, i)
        indent = "\t\t"
        code += indent + ".clka(clk),\n"
        code += indent + ".ena(bram_r_ena[%d]),\n"%(i)
        code += indent + ".wea(bram_r_wea[%d]),\n"%(i)
        code += indent + ".addra(bram_r_addra[%d]),\n"%(i)
        code += indent + ".dina(bram_r_dina[%d]),\n"%(i)
        code += indent + ".clkb(clk),\n"
        code += indent + ".enb(bram_r_enb[%d]),\n"%(i)
        code += indent + ".addrb(bram_r_addrb[%d]),\n"%(i)
        code += indent + ".doutb(bram_r_doutb[%d])\n"%(i)
        indent = "\t"
        code += indent + ");\n"

    # initial ps bram_a bram_b bram_r
    code += indent + "initial begin\n"
    indent = "\t\t"
    code += indent + "bram_en = 1;\n"
    code += indent + "bram_rst = 0;\n"
    code += indent + "bram_din = 0;\n"
    code += indent + "bram_addr = 0;\n"
    code += indent + "bram_we = 0;\n"
    code += indent + "dma_r0_tready = 0;\n"
    code += indent + "dma_r1_tready = 0;\n"
    code += indent + "dma_w0_tdata = 0;\n"
    code += indent + "dma_w0_tkeep = 0;\n"
    code += indent + "dma_w0_tlast = 0;\n"
    code += indent + "dma_w0_tvalid = 0;\n"
    for i in range(WEIGHT_BUF_COL):
        code += indent + "bram_a_ena[%d] = 1;\n"%(i)
        code += indent + "bram_a_wea[%d] = 0;\n"%(i)
        code += indent + "bram_a_addra[%d] = 0;\n"%(i)
        code += indent + "bram_a_dina[%d] = 0;\n"%(i)
        code += indent + "bram_a_enb[%d] = 1;\n"%(i)
        code += indent + "bram_a_addrb[%d] = 0;\n"%(i)
    for i in range(FEATURE_MAP_BUF_COL):
        code += indent + "bram_b_ena[%d] = 1;\n"%(i)
        code += indent + "bram_b_wea[%d] = 0;\n"%(i)
        code += indent + "bram_b_addra[%d] = 0;\n"%(i)
        code += indent + "bram_b_dina[%d] = 0;\n"%(i)
        code += indent + "bram_b_enb[%d] = 1;\n"%(i)
        code += indent + "bram_b_addrb[%d] = 0;\n"%(i)
    for i in range(OUTPUT_BUF_COL):
        code += indent + "bram_r_ena[%d] = 1;\n"%(i)
        code += indent + "bram_r_wea[%d] = 0;\n"%(i)
        code += indent + "bram_r_addra[%d] = 0;\n"%(i)
        code += indent + "bram_r_dina[%d] = 0;\n"%(i)
        code += indent + "bram_r_enb[%d] = 1;\n"%(i)
        code += indent + "bram_r_addrb[%d] = 0;\n"%(i)
    indent = "\t"
    code += indent + "end\n"



    code += "endmodule"

    return code