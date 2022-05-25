

import cpu_float
import cpu_int
import conv
import fc
import post_process


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
        divided_border,
        submatrix_size,
        calc_process,
        model_dir
    )
    
    # 生成conv

    # 生成fc

    # 生成post_process

    # 生成main

    # 生成c

    # 生成instrset