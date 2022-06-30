



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
import op


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

    
    # # instruction filed for add
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
    ps_total_count_for_add = max(math.ceil(math.log2(max_count)+1), 1)
    
    # # # layer mux
    add_amount = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Add or
            type(node) == op.QAdd):
            add_amount += 1
    ps_layer_mux_for_add = max(math.ceil(math.log2(add_amount)), 1)
    
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