



'''
instruction from ps to pl
1. calculation type(including:
    conv(including post_process), 
    fc,
    max_pool(not supported yet),
    avg_pool(not supported ye))
 - for conv and post_process
2. weight data length(in unit of bram row)
3. feature map data length(int unit of bram row)
4. instruction begin addr to execute
5. instruction end addr(addr of the instruction just after the last 
    instruction that should be executed)
6. terminator(in a new bram word)
 - for fully connect
2. activation
3. hidden_channel
4. output_channel
5. layer_mux
6. terminator
 - for add
2. total_count  (number of data to add)
3. layer_mux

instruction in pl
1. calculation type(including:
    conv,
    post_process,
    write_back,
    terminate)
 - for conv
2. multiply side length
3. A left side length
4. B up side length
5. weight buffer read start line
6. feature map buffer read start line
7. output buffer store start line
8. just store or accumulate
9. layer mux
 - for post_process
2. side len(up side len of result block, decide when to increase ppchannel)
3. start channel
4. layer mux
5. output buffer read start line
6. process lines(in unit of bram_C row)
7. activation
 - for write_back
2. write back rows
'''

import math
from typing import Type
from unicodedata import decimal

from add import decimal_to_bin
import analyser
import op
import util


def conv_in_graph(calculation_graph):
    '''
    Judge if there is any `conv` operators in the calculation_graph
    '''
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or 
            type(node) == op.QConv2d):
            return True
    return False


def fc_in_graph(calculation_graph):
    '''
    Judge if there is any `fc` operators in the calculation_graph
    '''
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Dense or
            type(node) == op.QDense):
            return True
    return False


def add_in_graph(calculation_graph):
    '''
    Judge if there is any `add` operators in the calculation_graph
    '''
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Add or 
            type(node) == op.QAdd):
            return True
    return False


def analyse_instr(
    analyse_result,
    calculation_graph
):
    '''
    Analyse the bit width need of each section in the instructions
    Attention: for matrix side lengths, since they are all power of 2, 
    we record them by exponent, and can save bits in this way.
    '''
    # read analyse_result
    resource_analyse_result = analyse_result["resource_analyse_result"]
    divided_border = analyse_result["divided_border"]
    submatrix_size = analyse_result["submatrix_size"]
    tensor_expr = analyse_result["tensor_expr"]
    calc_process = analyse_result["calc_process"]
    c_max_usage = analyse_result["c_max_usage"]
    calc_process_with_parallel = analyse_result["calc_process_with_parallel"]
    cost = analyse_result["cost"]
    bram_group = resource_analyse_result["bram_group"]
    bram_col_c_need_per_bram_group = resource_analyse_result[
        "bram_col_c_need_per_bram_group"]
    depth_c_need_per_bram_col = resource_analyse_result[
        "depth_c_need_per_bram_col"]
    total_bram_need = resource_analyse_result["total_bram_need"]
    bram_avaliable = resource_analyse_result["bram_avaliable"]
    max_len_support = resource_analyse_result["max_matrix_len_support"]
    min_len_support = resource_analyse_result["min_matrix_len_support"]
    calc_unit_per_bram_group = resource_analyse_result[
        "calc_unit_per_bram_group"]
    total_lut_need = resource_analyse_result["total_lut_need"]
    lut_avaliable = resource_analyse_result["lut_avaliable"]
    more_radical_allocation = resource_analyse_result[
        "more_radical_allocation"]

    # instruction from ps to pl

    # # instruction field for all ps instructions
    ps_calculation_type = 4


    # # instruction field for conv and post_process
    # # # weight data length(in unit of bram row)
    '''
    calculate weight_data_length according to the depth of bram_A
    (weight_data_length=mat_A_left_side*mat_A_up_side//max_len_support,
    so it must be power of 2)
    '''
    bram_depth = 512
    ps_weight_data_length_for_conv = math.ceil(math.log2(
        max_len_support))+1 if (bram_group > 0) else \
        math.ceil(math.log2(bram_depth))+1
    ps_weight_data_length_for_conv_exponent = math.ceil(math.log2(math.log2(
        max_len_support)))+1 if (bram_group > 0) else \
        math.ceil(math.log2(math.log2(bram_depth)))+1
    
    # # # feature map data length(int unit of bram row)
    '''
    calculate feature_map_data_length according to the depth of bram_B
    '''
    ps_feature_map_data_length_for_conv = math.ceil(math.log2(
        max_len_support))+1 if (bram_group > 0) else math.ceil(
        math.log2(bram_depth))+1
    ps_feature_map_data_length_for_conv_exponent = math.ceil(math.log2(
        math.log2(max_len_support)))+1 if (bram_group > 0) else math.ceil(
        math.log2(math.log2(bram_depth)))+1

    # # # instruction begin addr to execute
    '''
    count the number of calculation instructions in calc_process_with_parallel
    and multiply with 3 to be the estimation of the pl instruction amount
    '''
    calc_count = 0
    for layer in calc_process_with_parallel:
        for instr_pair in layer:
            instr_left = instr_pair[0]
            instr_right = instr_pair[1]
            for instr in instr_left:
                if("=" in instr):
                    calc_count += 1
            if(instr_right is not None):
                for instr in instr_right:
                    if("=" in instr):
                        calc_count += 1
    instr_amount_estimate = calc_count * 3
    ps_instr_begin_addr_for_conv = math.ceil(math.log2(instr_amount_estimate))

    # # # instruction end addr(addr of the instruction just after the last 
    # # #     instruction that should be executed))
    ps_instr_end_addr_for_conv = ps_instr_begin_addr_for_conv

    # # # total bit width need by ps conv post_process
    ps_bit_width_need_conv = ps_calculation_type + \
        ps_weight_data_length_for_conv_exponent + \
        ps_feature_map_data_length_for_conv_exponent + \
        ps_instr_begin_addr_for_conv + ps_instr_end_addr_for_conv


    # # instruction field for fully connect
    # # # activation
    ps_activation_for_fc = 4

    # # # hidded_channel
    '''
    find the max hidden channel in calcultion_graph and calculate the bit width
    according to it
    '''
    max_hidden_channel = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QDense):
            input_id = node.input
            input_node = calculation_graph[input_id]
            input_shape = input_node.output_shape
            hidden_channel = input_shape[1]
            max_hidden_channel = max(max_hidden_channel, hidden_channel)
    ps_hidden_channel_for_fc = math.ceil(math.log2(max_hidden_channel))+1

    # # # output_channel
    max_output_channel = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QDense):
            output_shape = node.output_shape
            output_channel = output_shape[1]
            max_output_channel = max(max_output_channel, output_channel)
    ps_output_channel_for_fc = math.ceil(math.log2(max_output_channel))+1

    # # # layer mux
    fc_amount = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Dense or
            type(node) == op.QDense):
            fc_amount += 1
    ps_layer_mux_for_fc = max(math.ceil(math.log2(fc_amount)), 1)
    
    # # # ps bit width need by fc
    ps_bit_width_need_fc = ps_calculation_type + \
        ps_activation_for_fc + ps_hidden_channel_for_fc + \
        ps_output_channel_for_fc + ps_layer_mux_for_fc

    
    # # instruction field for add
    # # # total_count
    max_count = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Add or
            type(node) == op.QAdd):
            output_shape = calculation_graph[node.input1].output_shape
            count = 1
            for i in output_shape:
                count *= i
            max_count = max_count if(max_count > count) else count
    ps_total_count_for_add = max(math.ceil(math.log2(max_count)+1), 1) if(
        add_in_graph(calculation_graph)) else 0
    
    # # # layer mux
    add_amount = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Add or
            type(node) == op.QAdd):
            add_amount += 1
    ps_layer_mux_for_add = max(math.ceil(math.log2(add_amount)), 1) if(
        add_in_graph(calculation_graph)) else 0
    
    # # # ps bit width need by add
    ps_bit_width_need_add = ps_calculation_type + \
        ps_total_count_for_add + ps_layer_mux_for_add
    
    # # ps bit width need
    ps_bit_width_need = math.ceil(
        max(
            ps_bit_width_need_conv, 
            ps_bit_width_need_fc,
            ps_bit_width_need_add
        ) / 32) * 32



    # instructions in pl

    # # instruction field for all pl instructions
    pl_calculation_type = 4

    # # instruction field for convolution
    # # # multiply side length
    '''
    find the max multiply side length and calculate the bit width
    according to it
    '''
    max_multiply_side_length = 0
    for layer_index, layer in enumerate(submatrix_size):
        submatrix_size_A = layer[0]
        submatrix_size_B = layer[1]
        for row in submatrix_size_A:
            for block in row:
                multiply_side_length = block[1]
                max_multiply_side_length = max(
                    max_multiply_side_length, multiply_side_length)
    pl_multiply_side_length_for_conv = math.ceil(
        math.log2(max_multiply_side_length))+1
    pl_multiply_side_length_for_conv_exponent = math.ceil(math.log2(
        math.log2(max_multiply_side_length)))+1

    # # # A left side length
    max_A_left_side_length = 0
    for layer_index, layer in enumerate(submatrix_size):
        submatrix_size_A = layer[0]
        submatrix_size_B = layer[1]
        for row in submatrix_size_A:
            for block in row:
                A_left_side_length = block[0]
                max_A_left_side_length = max(
                    max_A_left_side_length, A_left_side_length)
    pl_A_left_side_length_for_conv = math.ceil(
        math.log2(max_A_left_side_length))+1
    pl_A_left_side_length_for_conv_exponent = math.ceil(math.log2(
        math.log2(max_A_left_side_length)))+1
    
    # # # B up side length
    max_B_up_side_length = 0
    for layer_index, layer in enumerate(submatrix_size):
        submatrix_size_A = layer[0]
        submatrix_size_B = layer[1]
        for row in submatrix_size_B:
            for block in row:
                B_up_side_length = block[1]
                max_B_up_side_length = max(
                    max_B_up_side_length, B_up_side_length)
    pl_B_up_side_length_for_conv = math.ceil(
        math.log2(max_B_up_side_length))+1
    pl_B_up_side_length_for_conv_exponent = math.ceil(math.log2(
        math.log2(max_B_up_side_length)))+1
    
    # # # weight buffer read start line
    pl_weight_buffer_read_start_line_for_conv = math.ceil(math.log2(
        max_len_support)) if (bram_group > 0) else math.ceil(
        math.log2(bram_depth))
    
    # # # feature map buffer read start line
    pl_feature_map_buffer_read_start_line_for_conv = math.ceil(math.log2(
        max_len_support)) if (bram_group > 0) else math.ceil(
        math.log2(bram_depth))
    
    # # # output buffer store start line
    pl_output_buffer_store_start_line_for_conv = math.ceil(math.log2(
        depth_c_need_per_bram_col))
    
    # # # store or accumulate
    pl_store_or_accumulate_for_conv = 1

    # # # layer mux
    conv_amount = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or
            type(node) == op.QConv2d):
            conv_amount += 1
    pl_layer_mux_for_conv = max(math.ceil(math.log2(conv_amount)), 1)

    # # # pl bit width need by conv
    pl_bit_width_need_conv = pl_calculation_type + \
        pl_multiply_side_length_for_conv_exponent + \
        pl_A_left_side_length_for_conv_exponent + \
        pl_B_up_side_length_for_conv_exponent + \
        pl_weight_buffer_read_start_line_for_conv + \
        pl_feature_map_buffer_read_start_line_for_conv + \
        pl_output_buffer_store_start_line_for_conv + \
        pl_store_or_accumulate_for_conv + pl_layer_mux_for_conv


    # # instruction field for post process
    # # # side len
    max_result_block_up_side_length = 0
    for layer_index, layer in enumerate(submatrix_size):
        submatrix_size_A = layer[0]
        submatrix_size_B = layer[1]
        for row in submatrix_size_B:
            for block in row:
                result_block_up_side_length = block[1]
                max_result_block_up_side_length = max(
                    max_result_block_up_side_length, 
                    result_block_up_side_length)
    pl_side_len_for_pp = math.ceil(math.log2(
        max_result_block_up_side_length))+1
    pl_side_len_for_pp_exponent = math.ceil(math.log2(math.log2(
        max_result_block_up_side_length)))+1
    
    # # # start channel
    max_channel = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or
            type(node) == op.QConv2d):
            output_channel = node.output_channel
            max_channel = max(max_channel, output_channel)
    pl_start_channel_for_pp = math.ceil(math.log2(max_channel))
    
    # # # layer mux
    pl_layer_mux_for_pp = max(math.ceil(math.log2(conv_amount)), 1)
    
    # # # output buffer read start line
    pl_output_buffer_read_start_line_for_pp = math.ceil(math.log2(
        depth_c_need_per_bram_col))
    
    # # # process lines
    pl_process_lines_for_pp = math.ceil(math.log2(depth_c_need_per_bram_col))+1
    
    # # # activation
    pl_activation_for_pp = 4

    # # # pl bit width need by post process
    pl_bit_width_need_pp = pl_calculation_type + \
        pl_side_len_for_pp_exponent + \
        pl_start_channel_for_pp + pl_layer_mux_for_pp + \
        pl_output_buffer_read_start_line_for_pp + pl_process_lines_for_pp + \
        pl_activation_for_pp
    

    # # instruction field for write back
    # # # write back rows
    pl_write_back_rows_for_wb = math.ceil(math.log2(
        depth_c_need_per_bram_col))+1
    
    # # # pl bit width need by write back
    pl_bit_width_need_wb = pl_calculation_type + pl_write_back_rows_for_wb
    
    # # pl bit width need
    pl_bit_width_need = math.ceil(max(
        pl_bit_width_need_conv,
        pl_bit_width_need_pp,
        pl_bit_width_need_wb
    ) / 32) * 32
    
    

    # return result
    # marked by 'e' means the variable is recorded by exponent
    return {
        "ps_calculation_type": pl_calculation_type,
        "ps_weight_data_length_for_conv": ps_weight_data_length_for_conv, #e
        "ps_weight_data_length_for_conv_exponent": 
            ps_weight_data_length_for_conv_exponent,
        "ps_feature_map_data_length_for_conv": 
            ps_feature_map_data_length_for_conv, #e
        "ps_feature_map_data_length_for_conv_exponent":
            ps_feature_map_data_length_for_conv_exponent,
        "ps_instr_begin_addr_for_conv": ps_instr_begin_addr_for_conv,
        "ps_instr_end_addr_for_conv": ps_instr_end_addr_for_conv,
        "ps_bit_width_need_for_conv": ps_bit_width_need_conv,
        "ps_activation_for_fc": ps_activation_for_fc,
        "ps_hidden_channel_for_fc": ps_hidden_channel_for_fc,
        "ps_output_channel_for_fc": ps_output_channel_for_fc,
        "ps_layer_mux_for_fc": ps_layer_mux_for_fc,
        "ps_bit_width_need_for_fc": ps_bit_width_need_fc,
        "ps_total_count_for_add": ps_total_count_for_add,
        "ps_layer_mux_for_add": ps_layer_mux_for_add,
        "ps_bit_width_need_for_add": ps_bit_width_need_add,
        "ps_bit_width_need": ps_bit_width_need,
        "pl_calculation_type": pl_calculation_type,
        "pl_mult_side_length_for_conv": pl_multiply_side_length_for_conv, #e
        "pl_mult_side_length_for_conv_exponent": 
            pl_multiply_side_length_for_conv_exponent,
        "pl_A_left_side_length_for_conv": pl_A_left_side_length_for_conv, #e
        "pl_A_left_side_length_for_conv_exponent": 
            pl_A_left_side_length_for_conv_exponent,
        "pl_B_up_side_length_for_conv": pl_B_up_side_length_for_conv, #e
        "pl_B_up_side_length_for_conv_exponent": 
            pl_B_up_side_length_for_conv_exponent,
        "pl_weight_buffer_read_start_line_for_conv": 
            pl_weight_buffer_read_start_line_for_conv,
        "pl_feature_map_buffer_read_start_line_for_conv":
            pl_feature_map_buffer_read_start_line_for_conv,
        "pl_output_buffer_store_start_line_for_conv":
            pl_output_buffer_store_start_line_for_conv,
        "pl_store_or_accumulate_for_conv": pl_store_or_accumulate_for_conv,
        "pl_layer_mux_for_conv": pl_layer_mux_for_conv,
        "pl_bit_width_need_for_conv": pl_bit_width_need_conv,
        "pl_side_length_for_pp": pl_side_len_for_pp, #e
        "pl_side_length_for_pp_exponent": pl_side_len_for_pp_exponent,
        "pl_start_channel_for_pp": pl_start_channel_for_pp,
        "pl_layer_mux_for_pp": pl_layer_mux_for_pp,
        "pl_output_buffer_read_start_line_for_pp": 
            pl_output_buffer_read_start_line_for_pp,
        "pl_process_lines_for_pp": pl_process_lines_for_pp,
        "pl_activation_for_pp": pl_activation_for_pp,
        "pl_bit_width_need_for_pp": pl_bit_width_need_pp,
        "pl_write_back_rows_for_wb": pl_write_back_rows_for_wb,
        "pl_bit_width_need_for_wb": pl_bit_width_need_wb,
        "pl_bit_width_need": pl_bit_width_need
    }


def generate_instruction(
    calculation_graph,
    im2col_shape,
    analyse_result,
    instr_analyse_result,
    calc_process,
    calc_process_with_parallel,
):
    '''
    Generate instructions for instr_set.v
    '''

    # width of per data
    data_width = 8
    c_data_width = 32

    # list fields and bit width of conv, pp, wb instruction
    instr_width_for_conv = [
        ("pl_calculation_type", 
            instr_analyse_result["pl_calculation_type"]),
        ("pl_mult_side_length_for_conv_exponent", 
            instr_analyse_result["pl_mult_side_length_for_conv_exponent"]),
        ("pl_A_left_side_length_for_conv_exponent",
            instr_analyse_result["pl_A_left_side_length_for_conv_exponent"]),
        ("pl_B_up_side_length_for_conv_exponent",
            instr_analyse_result["pl_B_up_side_length_for_conv_exponent"]),
        ("pl_weight_buffer_read_start_line_for_conv",
            instr_analyse_result["pl_weight_buffer_read_start_line_for_conv"]),
        ("pl_feature_map_buffer_read_start_line_for_conv",
            instr_analyse_result[
                "pl_feature_map_buffer_read_start_line_for_conv"]),
        ("pl_output_buffer_store_start_line_for_conv",
            instr_analyse_result["pl_output_buffer_store_start_line_for_conv"]),
        ("pl_store_or_accumulate_for_conv",
            instr_analyse_result["pl_store_or_accumulate_for_conv"]),
        ("pl_layer_mux_for_conv",
            instr_analyse_result["pl_layer_mux_for_conv"]),
    ]
    pl_bit_width_need_for_conv = instr_analyse_result[
        "pl_bit_width_need_for_conv"]
    instr_width_for_pp = [
        ("pl_calculation_type", 
            instr_analyse_result["pl_calculation_type"]),
        ("pl_side_length_for_pp_exponent",
            instr_analyse_result["pl_side_length_for_pp_exponent"]),
        ("pl_start_channel_for_pp",
            instr_analyse_result["pl_start_channel_for_pp"]),
        ("pl_layer_mux_for_pp",
            instr_analyse_result["pl_layer_mux_for_pp"]),
        ("pl_output_buffer_read_start_line_for_pp",
            instr_analyse_result["pl_output_buffer_read_start_line_for_pp"]),
        ("pl_process_lines_for_pp",
            instr_analyse_result["pl_process_lines_for_pp"]),
        ("pl_activation_for_pp",
            instr_analyse_result["pl_activation_for_pp"]),
    ]
    pl_bit_width_need_for_pp = instr_analyse_result["pl_bit_width_need_for_pp"]
    instr_width_for_wb = [
        ("pl_calculation_type", 
            instr_analyse_result["pl_calculation_type"]),
        ("pl_write_back_rows_for_wb",
            instr_analyse_result["pl_write_back_rows_for_wb"]),
    ]
    pl_bit_width_need_for_wb = instr_analyse_result["pl_bit_width_need_for_wb"]
    pl_bit_width_need = instr_analyse_result["pl_bit_width_need"]

    # divide im2col matrix to submatrix
    divided_border = analyser.cut_im2col_matrix(
        im2col_shape,
        calculation_graph,
        analyse_result["resource_analyse_result"]["max_matrix_len_support"],
        analyse_result["resource_analyse_result"]["min_matrix_len_support"]
    )
    
    # get submatrix sizes 
    submatrix_size = analyser.get_submatrix_size(divided_border)

    '''
    This function only generates instructions for instr_set.v, that is to say:
    only generate instructions for pl self control calculations, including:
    conv2d, post_process, write_back.
    Since the calc instructions in calc_process_with_parallel only appears in
    the left line, we only need to focus on the left line
    '''
    '''
    The results should includes:
    1. The instructions in sequence
    2. The beginning index and the finishing index of each pair, indexing by
      layer_index and pair_index
    Caution: According to the current calc process plan rule, it does not 
    ensure that there must be a `store` just after a `load`, that is to say, 
    there may be two or more `load` connected
    '''
    '''
    How to generate instructions:
    0. Get buffer width and depth from analyse_result
    1. For a submatrix, get its width and height from `submatrix_size`
    2. Calc the matrix's size, and how many buffer lines it need
    3. According to the `load` instructions, infer each matrix's store 
      position, and save these data
    4. According to sequence that the `C` matrix appeared in the calc 
      instructions, add `C` matrix into output buffer sequentially, and infer 
      their position. (Attention: data width of C matrix is 32)
    5. Generate conv instructions for each `=, +=` action
    6. Go back and generate pp instructions for each C block
    7. Generate wb instructions for all `store` actions in a pair. (Maybe 
      need to check if the `C` matrices are continuously in the buffer)
    '''
    resource_analyse_result = analyse_result["resource_analyse_result"]
    max_len_support = resource_analyse_result["max_matrix_len_support"]
    bram_group = resource_analyse_result["bram_group"]
    buffer_A_width = max_len_support if(bram_group == 0) else \
        max_len_support * bram_group
    buffer_B_width = max_len_support if(bram_group == 0) else \
        max_len_support * bram_group
    bram_col_c_need_per_bram_group = resource_analyse_result[
        "bram_col_c_need_per_bram_group"]
    bram_col_c_need = bram_col_c_need_per_bram_group if(bram_group == 0) else \
        bram_col_c_need_per_bram_group * bram_group
    buffer_C_width = bram_col_c_need * 64 // c_data_width

    instructions = []       # generated instructions
    instruction_index = []  # instruction index for each pair
        # (each item: [(layer_index, pair_index), start_index, end_index])
    for layer_index, layer in enumerate(calc_process_with_parallel):
        submatrix_size_current_layer = submatrix_size[layer_index]
        submatrix_size_A = submatrix_size_current_layer[0]
        submatrix_size_B = submatrix_size_current_layer[1]
        # record buffer usage(each item: [block_name, start_line, end_line])
        buffer_A_usage = []
        buffer_B_usage = []
        buffer_C_usage = []
        # infer the activation type of this layer
        activation_type = 0
        current_node = analyser.search_conv2d(calculation_graph, layer_index+1)
        for node in calculation_graph:
            if(hasattr(node, "input")):
                if(node.input == current_node.id):
                    break
            elif(hasattr(node, "input1")):
                if(node.input1 == current_node.id):
                    break
            elif(hasattr(node, "input2")):
                if(node.input1 == current_node.id):
                    break
        if(type(node) == op.Relu or 
            type(node) == op.QRelu):
            activation_type = 1

        # traverse each layer in the calc_process
        for pair_index, process_pair in enumerate(layer):
            # overwrite buffer A and B in each new pair
            buffer_A_usage = []
            buffer_B_usage = []
            # traverse each process_pair in each layer
            left_line = process_pair[0]
            right_line = process_pair[1]
            # since we only generate instructions for pl self control calc, 
            # we only need to process `load, =, +=, store` series actions.
            # It should be mentioned that, we do not need to generate 
            # instructions for `load` action itself(`load` is controlled by 
            # ps), but since `load` is together with `=, +=`, we need to 
            # calculate bram usage according to `load` actions.
            action_need_to_process = False
            for action_index, action in enumerate(left_line):
                if(action[0:4] == "load" or
                    "=" in action or
                    action[0:5] == "store"):
                    action_need_to_process = True
            if(not action_need_to_process):
                continue

            # judge the type of the action group(`load, =` or `store`)
            action_type = None
            for action_index, action in enumerate(left_line):
                if(action[0:4] == "load" or 
                    "=" in action):
                    action_type = "load_calc"
                if(action[0:5] == "store"):
                    action_type = "store"
            
            def find_in_buffer_usage(buffer_usage, matrix):
                '''
                Find the matrix in the buffer_usage list
                '''
                for item in buffer_usage:
                    if(item[0] == matrix):
                        return item
                    
            # now process `load, =, +=, store`
            if(action_type == "load_calc"):
                for action_index, action in enumerate(left_line):
                    # allocate space for matrices in buffer
                    if("load" in action):
                        block = action.split(" ")[1]
                        matrix = block.split("_")[0]
                        row = int(block.split("_")[1])
                        col = int(block.split("_")[2])
                        if(matrix == "A"):
                            shape = submatrix_size_A[row][col]
                            size = shape[0] * shape[1]
                            buffer_line_need = size // buffer_A_width
                            start_line = 0
                            if(len(buffer_A_usage) > 0):
                                start_line = buffer_A_usage[-1][2] + 1
                            end_line = start_line + buffer_line_need - 1
                            buffer_A_usage.append([
                                block, start_line, end_line, shape
                            ])
                        elif(matrix == "B"):
                            shape = submatrix_size_B[row][col]
                            size = shape[0] * shape[1]
                            buffer_line_need = size // buffer_B_width
                            start_line = 0
                            if(len(buffer_B_usage) > 0):
                                start_line = buffer_B_usage[-1][2] + 1
                            end_line = start_line + buffer_line_need - 1
                            buffer_B_usage.append([
                                block, start_line, end_line, shape
                            ])
                        else:
                            raise TypeError("Unsupported matrix type")
                    elif("=" in action):
                        block = ""
                        if("+" in action):
                            block = action.split("+=")[0]
                        else:
                            block = action.split("=")[0]
                        matrix = block.split("_")[0]
                        row = int(block.split("_")[1])
                        col = int(block.split("_")[2])
                        shape = (submatrix_size_A[row][0][0], 
                            submatrix_size_B[0][col][1])
                        size = shape[0] * shape[1]
                        buffer_line_need = size // buffer_C_width
                        start_line = 0
                        if(len(buffer_C_usage) > 0):
                            start_line = buffer_C_usage[-1][2] + 1
                        end_line = start_line + buffer_line_need - 1
                        buffer_C_usage.append([
                            block, start_line, end_line, shape
                        ])
                for action_index, action in enumerate(left_line):
                    # generate conv instructions for `=, +=`
                    if(not "=" in action):
                        continue
                    save_type = 0
                    matrix_A = ""
                    matrix_B = ""
                    matrix_C = ""
                    if("+" in action):
                        save_type = 1
                        matrix_C = action.split("+=")[0]
                        matrix_A = action.split("+=")[1].split("*")[0]
                        matrix_B = action.split("+=")[1].split("*")[1]
                    else:
                        save_type = 0
                        matrix_C = action.split("=")[0]
                        matrix_A = action.split("=")[1].split("*")[0]
                        matrix_B = action.split("=")[1].split("*")[1]
                    C_information = find_in_buffer_usage(
                        buffer_C_usage, matrix_C)
                    A_information = find_in_buffer_usage(
                        buffer_A_usage, matrix_A)
                    B_information = find_in_buffer_usage(
                        buffer_B_usage, matrix_B)
                    instr = ""
                    instr_value_for_conv = {
                        "pl_calculation_type": 0,
                        "pl_mult_side_length_for_conv_exponent": 
                            math.ceil(math.log2(A_information[3][1])),
                        "pl_A_left_side_length_for_conv_exponent": 
                            math.ceil(math.log2(A_information[3][0])),
                        "pl_B_up_side_length_for_conv_exponent": 
                            math.ceil(math.log2(B_information[3][1])),
                        "pl_weight_buffer_read_start_line_for_conv": 
                            A_information[1],
                        "pl_feature_map_buffer_read_start_line_for_conv":
                            B_information[1],
                        "pl_output_buffer_store_start_line_for_conv":
                            C_information[1],
                        "pl_store_or_accumulate_for_conv": save_type,
                        "pl_layer_mux_for_conv": layer_index,
                    }
                    # generate for each field
                    for pair in instr_width_for_conv:
                        field = pair[0]
                        width = pair[1]
                        instr += decimal_to_bin(instr_value_for_conv[field], 
                            width)
                    # completion to 32n bit
                    instr += decimal_to_bin(0, pl_bit_width_need - 
                        pl_bit_width_need_for_conv)
                    instructions.append(instr)
                C_blocks = []
                for action_index, action in enumerate(left_line):
                    # find all C blocks in current process_pair
                    if(not "=" in action):
                        continue
                    matrix_C = ""
                    if("+" in action):
                        matrix_C = action.split("+=")[0]
                    else:
                        matrix_C = action.split("=")[0]
                    if(not matrix_C in C_blocks):
                        C_blocks.append(matrix_C)
                for block in C_blocks:
                    # generate pp instruction for each C block in current pair
                    C_information = find_in_buffer_usage(buffer_C_usage, block)
                    # infer C block channel according to A block
                    row = int(block.split("_")[1])
                    col = int(block.split("_")[2])
                    start_channel = 0
                    for i in range(row):
                        start_channel += submatrix_size_A[i][0][0]
                    instr = ""
                    instr_value_for_pp = {
                        "pl_calculation_type": 1,
                        "pl_side_length_for_pp_exponent": # up side of C block
                            math.ceil(math.log2(C_information[3][1])),
                        "pl_start_channel_for_pp": start_channel,
                        "pl_layer_mux_for_pp": layer_index,
                        "pl_output_buffer_read_start_line_for_pp":
                            C_information[1],
                        "pl_process_lines_for_pp": 
                            C_information[2] - C_information[1] + 1,
                        "pl_activation_for_pp": activation_type,
                    }
                    # generate for each field
                    for pair in instr_width_for_pp:
                        field = pair[0]
                        width = pair[1]
                        instr += decimal_to_bin(instr_value_for_pp[field], 
                            width)
                    # completion to 32n bit
                    instr += decimal_to_bin(0, pl_bit_width_need - 
                        pl_bit_width_need_for_pp)
                    instructions.append(instr)
                # record instructions index of current process_pair
                start_index = 0
                if(len(instruction_index) > 0):
                    start_index = instruction_index[-1][2] + 1
                instruction_count = len(instructions) - start_index
                end_index = start_index + instruction_count - 1
                instruction_index.append([(layer_index, pair_index), 
                    start_index, end_index])
            elif(action_type == "store"):
                # generate wb instructions for `store` actions
                start_end_lines = []
                for action_index, action in enumerate(left_line):
                    if(not "store" in action):
                        raise TypeError("Unsupported action type")
                    block = action.split(" ")[1]
                    information = find_in_buffer_usage(buffer_C_usage, block)
                    start_line = information[1]
                    end_line = information[2]
                    start_end_lines.append((start_line, end_line))
                # splice the blocks in start_end_lines
                i = 0
                j = i + 1
                while(True):
                    if(len(start_end_lines) <= 1):
                        break
                    if(i == len(start_end_lines)):
                        break
                    if(j == len(start_end_lines)):
                        i += 1
                        j = 0
                        continue
                    block1 = start_end_lines[i]
                    block2 = start_end_lines[j]
                    start1 = block1[0]
                    end1 = block1[1]
                    start2 = block2[0]
                    end2 = block2[1]
                    if(start2 == end1 + 1):
                        start_end_lines[i] = (start1, end2)
                        start_end_lines.remove(start_end_lines[j])
                        i = 0
                        j = i + 1
                    else:
                        j += 1
                # generate wb instructions for each block in start_end_lines
                for block in start_end_lines:
                    instr = ""
                    instr_value_for_wb = {
                        "pl_calculation_type": 2,
                        "pl_write_back_rows_for_wb":
                            block[1] - block[0],
                    }
                    # generate for each field
                    for pair in instr_width_for_wb:
                        field = pair[0]
                        width = pair[1]
                        instr += decimal_to_bin(instr_value_for_wb[field], 
                            width)
                    # completion to 32n bit
                    instr += decimal_to_bin(0, pl_bit_width_need - 
                        pl_bit_width_need_for_wb)
                    instructions.append(instr)
                # record instructions index of current process_pair
                start_index = 0
                if(len(instruction_index) > 0):
                    start_index = instruction_index[-1][2] + 1
                instruction_count = len(instructions) - start_index
                end_index = start_index + instruction_count - 1
                instruction_index.append([(layer_index, pair_index), 
                    start_index, end_index])
            else:
                raise TypeError("Unsupported action type")
    return instructions, instruction_index


def generate_instrset(
    MODULE_NAME,
    ADDR_WIDTH,
    INSTR_WIDTH,
    INSTRUCTIONS,
    DEBUG=True,
):
    '''
    Generate instrset.v
    '''

    # signals need to generate debug probes
    debug_signals = []

    code = ""

    # generate module
    code += "module %s(\n"%(MODULE_NAME)

    # generate ports
    indent = "\t"
    code += indent + "input clk,\n"
    code += indent + "input [%d:0] addr,\n"%(ADDR_WIDTH-1)
    code += indent + "output [%d:0] dout\n"%(INSTR_WIDTH-1)
    code += ");\n"

    # generate dataset  
    # TODO: here we round up the depth of dataset to 2**n, instead of the 
    # TODO: accurate depth, and may cost more LUTRAM(I dont know)
    code += indent + "reg [%d:0] instruction_set[%d:0];\n"%(INSTR_WIDTH-1,
        2**math.ceil(math.log2(len(INSTRUCTIONS)))-1)

    # assign dout
    code += indent + "assign dout = instruction_set[addr];\n"

    # initial instructions
    code += indent + "initial begin\n"
    indent = "\t\t"
    for index, instr in enumerate(INSTRUCTIONS):
        code += indent + "instruction_set[%d] = %d'b%s;\n"%(index, INSTR_WIDTH,
            instr)

    indent = "\t"
    code += indent + "end\n"


    code += "endmodule"

    return code