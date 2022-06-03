


from distutils.util import strtobool
import math


def decimal_to_one_hot_binary(value, width):
    '''
    tranfer decimal value to binary one-hot code with 'width' width
    '''
    code = ""
    for i in range(value):
        code = "0" + code
    code = "1" + code
    for i in range(width - value - 1):
        code = "0" + code
    return code


def gen_top(
    WEIGHT_BUF_COL: int,        # number of weight buffer columns
    WEIGUT_BUF_DEPTH: int,      # depth of weight buffer
    FEATURE_MAP_BUF_COL: int,   # number of feature map buffer columns
    FEATURE_MAP_BUF_DEPTH: int, # depth of feature map buffer
    OUTPUT_BUF_COL: int,        # number of output buffer columns
    OUTPUT_BUF_DEPTH: int,      # depth of output buffer
    MAX_LEN_SUPPORT: int,       # max len support
    CONV_MUX_WIDTH: int,        # conv layer mux width
    CALC_UNIT_PER_BRAM_GROUP: int,  # calc unit per bram group
    CONV_OUTPUT_PORTS: int,     # conv output ports
    FC_HIDDEN_LEN: int,         # max hidden len of fc layers
    FC_OUTPUT_LEN: int,         # max output len of fc layers
    FC_MUX_WIDTH: int,          # fc layer mux width
    PP_CHANNEL: int,            # max channel of conv
    PP_MUX_WIDTH: int,          # pp layer mux width
    INSTR_ANALYSE_RESULT: dict, # instructions' bit width dict
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


    # signals of modules
    # signals of conv
    for unit in range(CALC_UNIT_PER_BRAM_GROUP):
        code += indent + "reg [%d:0] conv_mux_%d;\n"%(CONV_MUX_WIDTH-1, unit)
        code += indent + "reg conv_in_valid_%d;\n"%(unit)
        code += indent + "reg [%d:0] conv_ina_%d;\n"%(
            WEIGHT_BUF_COL*8*8-1, unit)
        code += indent + "reg [%d:0] conv_inb_%d;\n"%(
            FEATURE_MAP_BUF_COL*8*8-1, unit)
        for port in CONV_OUTPUT_PORTS:
            if(port == MAX_LEN_SUPPORT):
                code += indent + "wire [31:0] conv_add%d_%d;\n"%(port, unit)
                code += indent + "wire conv_add%d_valid_%d;\n"%(port, unit)
                continue
            origin_width = 8
            use_width = origin_width + 1    # unsign extend width
            mult_width = use_width * 2      # with after mult
            accu = 1
            width = mult_width
            while(accu < port):
                accu *= 2
                width += 1
            total_width = (MAX_LEN_SUPPORT // port) * width
            code += indent + "wire [%d:0] conv_add%d_%d;\n"%(
                total_width, port, unit)
            code += indent + "wire conv_add%d_valid_%d;\n"%(port, unit)
        
    # signals of fc
    dma_bit_width = 64
    hidden_len_number = math.ceil(FC_HIDDEN_LEN / 8)
    hidden_len_width = math.ceil(math.ceil(math.log2(hidden_len_number)) / 8) \
        * 8
    output_len_width = math.ceil(math.ceil(math.log2(FC_OUTPUT_LEN)) / 8) * 8
    fc_activation_width = INSTR_ANALYSE_RESULT["ps_activation_for_fc"]
    code += indent + "reg [%d:0] fc_din;\n"%(dma_bit_width-1)
    code += indent + "reg fc_invalid;\n"
    code += indent + "reg fc_datatype;\n"
    code += indent + "reg [%d:0] fc_hidden_len;\n"%(hidden_len_width-1)
    code += indent + "reg [%d:0] fc_output_len;\n"%(output_len_width-1)
    code += indent + "reg [%d:0] fc_mux;\n"%(FC_MUX_WIDTH-1)
    code += indent + "reg [%d:0] fc_activation;\n"%(fc_activation_width-1)
    code += indent + "wire [7:0] fc_dout;\n"
    code += indent + "wire fc_outvalid;\n"

    # signals of post_process
    pp_activation_width = INSTR_ANALYSE_RESULT["pl_activation_for_pp"]
    for unit in range(CALC_UNIT_PER_BRAM_GROUP):
        output_buf_col_per_calc_unit = OUTPUT_BUF_COL // \
            CALC_UNIT_PER_BRAM_GROUP
        pp_channel_width = math.ceil(math.ceil(math.log2(PP_CHANNEL)) / 8) * 8
        code += indent + "reg [%d:0] pp_din_%d;\n"%(
            output_buf_col_per_calc_unit * 64 - 1, unit)
        code += indent + "reg [%d:0] pp_channel_%d;\n"%(
            pp_channel_width-1, unit)
        code += indent + "reg [%d:0] pp_mux_%d;\n"%(PP_MUX_WIDTH-1, unit)
        code += indent + "reg [%d:0] pp_activation_%d;\n"%(
            pp_activation_width-1, unit)
        code += indent + "wire [%d:0] pp_dout_%d;\n"%(
            output_buf_col_per_calc_unit * 16 - 1, unit)

    # signals of instrset
    pl_instr_addr_width = INSTR_ANALYSE_RESULT["ps_instr_begin_addr_for_conv"]
    pl_instr_width = INSTR_ANALYSE_RESULT["pl_bit_width_need"]
    code += indent + "reg [%d:0] instrset_addr;\n"%(pl_instr_addr_width-1)
    code += indent + "wire [%d:0] instrset_dout;\n"%(pl_instr_width-1)

    # initialize signals of modules
    for unit in range(CALC_UNIT_PER_BRAM_GROUP):
        code += indent + "conv_mux_%d = 0;\n"%(unit)
        code += indent + "conv_in_valid_%d = 0;\n"%(unit)
        code += indent + "conv_ina_%d = 0;\n"%(unit)
        code += indent + "conv_inb_%d = 0;\n"%(unit)
    code += indent + "fc_din = 0;\n"
    code += indent + "fc_invalid = 0;\n"
    code += indent + "fc_datatype = 0;\n"
    code += indent + "fc_hidden_len = 0;\n"
    code += indent + "fc_output_len = 0;\n"
    code += indent + "fc_mux = 0;\n"
    code += indent + "fc_activation = 0;\n"
    for unit in range(CALC_UNIT_PER_BRAM_GROUP):
        code += indent + "pp_din_%d = 0;\n"%(unit)
        code += indent + "pp_channel_%d = 0;\n"%(unit)
        code += indent + "pp_mux_%d = 0;\n"%(unit)
        code += indent + "pp_activation_%d = 0;\n"%(unit)
    code += indent + "instrset_addr = 0;\n"


    # signals for control
    # signals for instructions from ps
    ps_instr_width = INSTR_ANALYSE_RESULT["ps_bit_width_need"]
    ps_calc_type_width = INSTR_ANALYSE_RESULT["ps_calculation_type"]
    ps_weight_len_width = INSTR_ANALYSE_RESULT[
        "ps_weight_data_length_for_conv"]
    ps_feature_map_len_width = INSTR_ANALYSE_RESULT[
        "ps_feature_map_data_length_for_conv"]
    ps_instr_begin_addr_width = INSTR_ANALYSE_RESULT[
        "ps_instr_begin_addr_for_conv"]
    ps_instr_end_addr_width = INSTR_ANALYSE_RESULT[
        "ps_instr_end_addr_for_conv"]
    code += indent + "reg [%d:0] ps_instruction;\n"%(ps_instr_width-1)
    code += indent + "reg [%d:0] ps_calc_type;\n"%(ps_calc_type_width-1)
    code += indent + "reg [%d:0] ps_weight_len;\n"%(ps_weight_len_width-1)
    code += indent + "reg [%d:0] ps_feature_map_len;\n"%(
        ps_feature_map_len_width-1)
    code += indent + "reg [%d:0] ps_instr_start_addr;\n"%(
        ps_instr_begin_addr_width-1)
    code += indent + "reg [%d:0] ps_instr_end_addr;\n"%(
        ps_instr_end_addr_width-1)
    
    # signals for instructions from pl
    pl_instr_width = INSTR_ANALYSE_RESULT["pl_bit_width_need"]
    pl_calc_type_width = INSTR_ANALYSE_RESULT["pl_calculation_type"]
    code += indent + "reg [%d:0] pl_instruction;\n"%(pl_instr_width-1)
    code += indent + "reg [%d:0] pl_calc_type;\n"%(pl_calc_type_width-1)
    # signals for convolution instruction
    pl_mult_side_len_conv_width = INSTR_ANALYSE_RESULT[
        "pl_mult_side_length_for_conv"]
    pl_A_left_side_len_conv_width = INSTR_ANALYSE_RESULT[
        "pl_A_left_side_length_for_conv"]
    pl_B_up_side_len_conv_width = INSTR_ANALYSE_RESULT[
        "pl_B_up_side_length_for_conv"]
    pl_weight_start_addr_conv_width = INSTR_ANALYSE_RESULT[
        "pl_weight_buffer_read_start_line_for_conv"]
    pl_feature_map_start_addr_conv_width = INSTR_ANALYSE_RESULT[
        "pl_feature_map_buffer_read_start_line_for_conv"]
    pl_output_start_addr_conv_width = INSTR_ANALYSE_RESULT[
        "pl_output_buffer_store_start_line_for_conv"]
    pl_save_type_conv_width = INSTR_ANALYSE_RESULT[
        "pl_store_or_accumulate_for_conv"]
    pl_layer_mux_conv_width = INSTR_ANALYSE_RESULT["pl_layer_mux_for_conv"]
    code += indent + "reg [%d:0] pl_mult_side_len_conv;\n"%(
        pl_mult_side_len_conv_width-1)
    code += indent + "reg [%d:0] pl_A_left_side_len_conv;\n"%(
        pl_A_left_side_len_conv_width-1)
    code += indent + "reg [%d:0] pl_B_up_side_len_conv;\n"%(
        pl_B_up_side_len_conv_width-1)
    code += indent + "reg [%d:0] pl_weight_start_addr_conv;\n"%(
        pl_weight_start_addr_conv_width-1)
    code += indent + "reg [%d:0] pl_feature_map_start_addr_conv;\n"%(
        pl_feature_map_start_addr_conv_width-1)
    code += indent + "reg [%d:0] pl_output_start_addr_conv;\n"%(
        pl_output_start_addr_conv_width-1)
    code += indent + "reg [%d:0] pl_save_type_conv;\n"%(
        pl_save_type_conv_width-1)
    code += indent + "reg [%d:0] pl_mux_conv;\n"%(pl_layer_mux_conv_width-1)
    # signals for post_process instructions
    pl_side_len_pp_width = INSTR_ANALYSE_RESULT["pl_side_length_for_pp"]
    pl_start_channel_pp_width = INSTR_ANALYSE_RESULT["pl_start_channel_for_pp"]
    pl_layer_mux_pp_width = INSTR_ANALYSE_RESULT["pl_layer_mux_for_pp"]
    pl_output_start_addr_pp_width = INSTR_ANALYSE_RESULT[
        "pl_output_buffer_read_start_line_for_pp"]
    pl_process_lines_pp_width = INSTR_ANALYSE_RESULT["pl_process_lines_for_pp"]
    pl_activation_pp_width = INSTR_ANALYSE_RESULT["pl_activation_for_pp"]
    code += indent + "reg [%d:0] pl_side_len_pp;\n"%(pl_side_len_pp_width-1)
    code += indent + "reg [%d:0] pl_start_channel_pp;\n"%(
        pl_start_channel_pp_width-1)
    code += indent + "reg [%d:0] pl_mux_pp;\n"%(pl_layer_mux_pp_width-1)
    code += indent + "reg [%d:0] pl_output_start_addr_pp;\n"%(
        pl_output_start_addr_pp_width-1)
    code += indent + "reg [%d:0] pl_process_lines_pp;\n"%(
        pl_process_lines_pp_width-1)
    code += indent + "reg [%d:0] pl_activation_pp;\n"%(
        pl_activation_pp_width-1)
    # signals for write back instructions
    pl_write_len_wb_width = INSTR_ANALYSE_RESULT["pl_write_back_rows_for_wb"]
    code += indent + "reg [%d:0] pl_write_len_wb;\n"%(pl_write_len_wb_width-1)
    # signals for fc
    ps_activation_fc_width = INSTR_ANALYSE_RESULT["ps_activation_for_fc"]
    ps_hidden_channel_fc_width = INSTR_ANALYSE_RESULT[
        "ps_hidden_channel_for_fc"]
    ps_output_channel_fc_width = INSTR_ANALYSE_RESULT[
        "ps_output_channel_for_fc"]
    ps_layer_mux_fc_width = INSTR_ANALYSE_RESULT["ps_layer_mux_for_fc"]
    code += indent + "reg [%d:0] ps_activation_fc;\n"%(
        ps_activation_fc_width-1)
    code += indent + "reg [%d:0] ps_hidden_channel_fc;\n"%(
        ps_hidden_channel_fc_width-1)
    code += indent + "reg [%d:0] ps_output_channel_fc;\n"%(
        ps_output_channel_fc_width-1)
    code += indent + "reg [%d:0] ps_layer_mux_fc;\n"%(ps_layer_mux_fc_width-1)

    # signals for top state machine
    states = [
        "PS_INSTR_READ",
        "PL_INSTR_READ",
        "CONVOLUTION",
        "POST_PROCESS",
        "WRITE_BACK",
        "FULLY_CONNECT",
        "DATA_TRANSFER",
        "PL_INSTR_FINISH",
    ]
    state_bit_width = len(states)
    code += indent + "reg [%d:0] state;\n"%(state_bit_width-1)
    for n, state in enumerate(states):
        code += indent + "parameter [%d:0] %s = %d'b%s;\n"%(state_bit_width-1, 
            state, state_bit_width,
            decimal_to_one_hot_binary(n, state_bit_width))
    code += indent + "reg [7:0] internal_state;\n"
    code += indent + "reg [7:0] dma_state;\n"
    print(CONV_OUTPUT_PORTS)




    code += "endmodule"

    return code