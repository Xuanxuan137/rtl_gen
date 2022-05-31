

import math
import op
import cpu_float
import cpu_int
import conv
import fc
import post_process
import top
import instr


def get_conv_amount(calculation_graph):
    '''
    Get the amount of conv layers in calculation_graph
    '''
    count = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or 
            type(node) == op.QConv2d):
            count += 1
    return count


def get_fc_amount(calculation_graph):
    '''
    Get the amount of fc layers in calculation_graph
    '''
    count = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Dense or
            type(node) == op.QDense):
            count += 1
    return count


def get_output_ports(submatrix_size):
    '''
    Get the output_ports type that the conv module need
    by counting the length of the multiply sides of A and B, 
    and return the list of the output ports
    '''
    output_ports = []
    for layer_index, layer in enumerate(submatrix_size):
        submatrix_size_A = layer[0]
        submatrix_size_B = layer[1]
        for row in submatrix_size_A:
            for block in row:
                multiply_side = block[1]
                if(not multiply_side in output_ports):
                    output_ports.append(multiply_side)
    output_ports.sort()
    return output_ports


def get_conv_zero(calculation_graph):
    '''
    Get zero_x zero_w zero_b zero_y list of conv2d
    '''
    zero_x_list = []
    zero_w_list = []
    zero_b_list = []
    zero_y_list = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            zero_x_list.append(node.zero_x)
            zero_w_list.append(node.zero_w)
            zero_b_list.append(node.zero_b)
            zero_y_list.append(node.zero_y)
    return zero_x_list, zero_w_list, zero_b_list, zero_y_list


def get_hidden_len(calculation_graph):
    '''
    Get the hidden layer(input layer) length of each fc layer
    '''
    hidden_len = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Dense or
            type(node) == op.QDense):
            input_id = node.input
            input_node = calculation_graph[input_id]
            input_shape = input_node.output_shape
            input_len = input_shape[1]
            hidden_len.append(input_len)
    return hidden_len


def get_output_len(calculation_graph):
    '''
    Get the output length of each fc layer
    '''
    output_len = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Dense or 
            type(node) == op.QDense):
            output_shape = node.output_shape
            output_len.append(output_shape[1])
    return output_len


def get_fc_bias(calculation_graph):
    '''
    Get the bias of each fc layer
    '''
    bias = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Dense or 
            type(node) == op.QDense):
            bias.append(node.bias)
    return bias


def get_fc_coe(calculation_graph):
    '''
    Get the coes of each fc layer
    '''
    coe = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QDense):
            coe.append(node.coe)
    return coe


def get_fc_rshift(calculation_graph):
    '''
    Get the rshift of each fc layer
    '''
    rshift = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QDense):
            rshift.append(node.rshift)
    return rshift


def get_fc_zero(calculation_graph):
    '''
    Get zero_x, zero_w, zero_b, zero_y of each fc layer
    '''
    zero_x_list = []
    zero_w_list = []
    zero_b_list = []
    zero_y_list = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QDense):
            zero_x_list.append(node.zero_x)
            zero_w_list.append(node.zero_w)
            zero_b_list.append(node.zero_b)
            zero_y_list.append(node.zero_y)
    return zero_x_list, zero_w_list, zero_b_list, zero_y_list


def get_fc_qmin_qmax(calculation_graph):
    '''
    Get qmin, qmax of each fc layer
    '''
    qmin_list = []
    qmax_list = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QDense):
            qmin_list.append(node.qmin)
            qmax_list.append(node.qmax)
    return qmin_list, qmax_list


def get_conv_bias(calculation_graph):
    '''
    Get bias of each conv layer
    '''
    bias = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or
            type(node) == op.QConv2d):
            bias.append(node.bias)
    return bias


def get_conv_coe(calculation_graph):
    '''
    Get coe of each conv layer
    '''
    coe = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            coe.append(node.coe)
    return coe


def get_conv_rshift(calcution_graph):
    '''
    Get rshift of each conv layer
    '''
    rshift = []
    for n, node in enumerate(calcution_graph):
        if(type(node) == op.QConv2d):
            rshift.append(node.rshift)
    return rshift


def get_conv_qmin_qmax(calculation_graph):
    '''
    Get qmin qmax of each conv layer
    '''
    qmin = []
    qmax = []
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            qmin.append(node.qmin)
            qmax.append(node.qmax)
    return qmin, qmax


def gen_code(
    analyse_result,
    data_on_chip,
    calculation_graph,
    im2col_shape,
    model_dir,
    float_mdoel_dir
):
    '''
    生成代码
    '''
    # 读取分析结果
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


    # # 生成cpu浮点计算代码
    cpu_float.gen_code(
        im2col_shape,
        divided_border,
        submatrix_size,
        calc_process,
        float_mdoel_dir
    )

    # 生成cpu整数计算代码
    cpu_int.gen_code(
        im2col_shape,
        calculation_graph,
        divided_border,
        submatrix_size,
        calc_process,
        model_dir
    )
    
    # 生成conv
    dma_width = 64  # bits
    conv_data_width = 8  # bits
    conv_amount = get_conv_amount(calculation_graph)
    conv_mux_width = max(math.ceil(math.log2(conv_amount)), 1)
    conv_output_ports = get_output_ports(submatrix_size)
    conv_zero_x, conv_zero_w, _, _ = get_conv_zero(calculation_graph)
    conv_code = conv.gen_conv(
        MODULE_NAME="conv",
        MUX_WIDTH=conv_mux_width,
        DATA_WIDTH=conv_data_width,
        DATA_NUMBER=max_len_support,
        OUTPUT_PORTS=conv_output_ports,
        ZERO_X=conv_zero_x,
        ZERO_W=conv_zero_w,
        DEBUG=True
    )
    with open("output/conv.v", "w") as f:
        f.write(conv_code)

    # 生成fc
    fc_data_width = 8
    fc_amount = get_fc_amount(calculation_graph)
    fc_mux_width = max(math.ceil(math.log2(fc_amount)), 1)
    fc_data_number = dma_width // fc_data_width
    fc_hidden_len = max(get_hidden_len(calculation_graph))
    fc_output_len = max(get_output_len(calculation_graph))
    fc_bias = get_fc_bias(calculation_graph)
    fc_coe = get_fc_coe(calculation_graph)
    fc_rshift = get_fc_rshift(calculation_graph)
    fc_zero_x, fc_zero_w, _, fc_zero_y = get_fc_zero(calculation_graph)
    fc_qmin, fc_qmax = get_fc_qmin_qmax(calculation_graph)
    fc_code = fc.gen_fc(
        MODULE_NAME="fc",
        MUX_WIDTH=fc_mux_width,
        DATA_WIDTH=fc_data_width,
        DATA_NUMBER=fc_data_number,
        HIDDEN_LEN=fc_hidden_len,
        OUTPUT_LEN=fc_output_len,
        BIAS=fc_bias,
        COE=fc_coe,
        RSHIFT=fc_rshift,
        ZERO_X=fc_zero_x,
        ZERO_W=fc_zero_w,
        ZERO_Y=fc_zero_y,
        QMAX=fc_qmax[0],
        DEBUG=True
    )
    with open("output/fc.v", "w") as f:
        f.write(fc_code)

    # 生成post_process
    pp_mux_width = max(math.ceil(math.log2(conv_amount)), 1)
    pp_data_width = 32
    bram_col_c_need = bram_col_c_need_per_bram_group if(bram_group == 0) else \
        bram_col_c_need_per_bram_group * bram_group
    bram_col_data_width = 64
    pp_data_number = bram_col_c_need * bram_col_data_width // pp_data_width
    pp_out_data_width = 8
    conv_bias = get_conv_bias(calculation_graph)
    conv_coe = get_conv_coe(calculation_graph)
    conv_rshift = get_conv_rshift(calculation_graph)
    _, _, _, conv_zero_y = get_conv_zero(calculation_graph)
    conv_qmin, conv_qmax = get_conv_qmin_qmax(calculation_graph)
    pp_code = post_process.gen_post_process(
        MODULE_NAME="post_process",
        MUX_WIDTH=pp_mux_width,
        DATA_WIDTH=pp_data_width,
        DATA_NUMBER=pp_data_number,
        OUT_DATA_WIDTH=pp_out_data_width,
        BIAS=conv_bias,
        COE=conv_coe,
        RSHIFT=conv_rshift,
        ZERO_Y=conv_zero_y,
        QMAX=conv_qmax[0],
        DEBUG=True
    )
    with open("output/post_process.v", "w") as f:
        f.write(pp_code)

    # 分析指令各部分需要的位宽
    instr_analyse_result = instr.analyse_instr(
        analyse_result,
        calculation_graph
    )

    # 生成main

    # 生成c

    # 生成instrset