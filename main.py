

import math
import argparse

import numpy as np

import conv
import fc
import post_process
import graph
import analyser
from util import *




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RTL")

    parser.add_argument("-m", "--model_dir", required=True, 
        help="Input model directory")
    parser.add_argument("-p", "--part", required=True, 
        help="Project part, e.g. xc7z020clg400-1")
    parser.add_argument("--LUT", type=int, help="The number of LUT")
    parser.add_argument("--FF", type=int, help="The number of Flip Flop")
    parser.add_argument("--BRAM", type=int, help="The number of BRAM")
    parser.add_argument("--DSP", type=int, help="The number of DSP")
    parser.add_argument("--BRAM_threshold", type=float, default=0.9, 
        help="BRAM usage threshold")
    parser.add_argument("--LUT_threshold", type=float, default=0.7, 
        help="LUT usage threshold")
    parser.add_argument("--data_on_chip", type=bool, default=True, 
        help="Put part of data on chip")
    parser.add_argument("--try_increase_c_bandwidth", type=bool, default=True,
        help="Try increase C bandwidth to improve performance, " \
            "but may cause overuse of bram")
    parser.add_argument("--optimize", type=int, default=2, help=
        "Try to increase performance. May not effective")

    args = parser.parse_args()
    
    clear_log()

    # 提取参数
    model_dir = args.model_dir
    project_part = args.part
    lut = args.LUT
    ff = args.FF
    bram = args.BRAM
    dsp = args.DSP
    bram_threshold = args.BRAM_threshold
    lut_threshold = args.LUT_threshold
    data_on_chip = args.data_on_chip
    try_increase_c_bandwidth = args.try_increase_c_bandwidth
    optimize = args.optimize
    xxlog("Read model_dir: %s"%(model_dir))
    xxlog("Read project_part: %s"%(project_part))
    xxlog("Read lut: %s"%(lut))
    xxlog("Read ff: %s"%(ff))
    xxlog("Read bram: %s"%(bram))
    xxlog("Read dsp: %s"%(dsp))
    xxlog("Read bram usage threshold: %s"%(bram_threshold))
    xxlog("Read lut usage threshold: %s"%(lut_threshold))
    xxlog("Read data_on_chip: %s"%(data_on_chip))
    xxlog("Read try increase C bandwidth: %s"%(try_increase_c_bandwidth))
    xxlog("Read optimize: %d"%(optimize))


    known_parts = [
        "xc7z020clg400-1", "xc7z020clg400-2", "xc7z020clg400-3",
    ]

    if(project_part not in known_parts):
        if(lut is None or
           ff is None or
           bram is None or
           dsp is None):
            xxlog("Since part is Unknown, you must specify the number of " \
                "LUT, FF, BRAM, DSP", XXError())
            raise ValueError("Since part is Unknown, you must specify " \
                "the number of LUT, FF, BRAM, DSP")
        
    if(data_on_chip == False):
        raise ValueError("data_on_chip=False is not supported yet")
    
    # 读取计算图
    calculation_graph = graph.read_calculation_graph(model_dir)

    # 如果未给定资源数量，则根据part设定资源数量
    lut, ff, bram, dsp = analyser.set_resources(
        project_part,
        lut,
        ff,
        bram,
        dsp
    )

    # 推算im2col后矩阵尺寸
    im2col_shape = analyser.infer_im2col_shape(calculation_graph)

    # 第一次资源分析
    first_analyse_result = analyser.analyse_resources_first_time(
        project_part,
        lut,
        ff,
        bram,
        dsp,
        bram_threshold,
        lut_threshold,
        try_increase_c_bandwidth,
        optimize,
        im2col_shape,
        calculation_graph
    )
    '''
    first_analyse_result = {
        # bram组数(组边长小于512时，此值为0)
        "bram_group": bram_group,   
        # 每组bram中C的列数
        "bram_col_c_need_per_bram_group": bram_col_C_need_per_bram_group, 
        # C需要的bram深度  
        "depth_c_need_per_bram_col": depth_C_need_per_bram_col,     
        # 所需要的总的bram数
        "total_bram_need": total_bram_need,     
        # 可用的bram数
        "bram_avaliable": int(bram_threshold*bram),
        # 支持的最大矩阵(也即bram组的边长)
        "max_matrix_len_support": max_len_support,  
        # 支持的最小矩阵
        "min_matrix_len_support": min_len_support,  
        # 每组bram分配的计算单元组数
        "calc_unit_per_bram_group": calc_uint_per_bram_group,   
        # 需要的总lut数
        "total_lut_need": total_lut_need,           
        # 可用的lut数
        "lut_avaliable": int(lut_threshold*lut)     
    }
    '''
    
    # 第一次拆分张量表达式
    first_tensor_expression = analyser.split_tensor_expression_first_time(
        first_analyse_result,
        im2col_shape,
        calculation_graph
    )
    '''
    return {
        # C是否被占满
        "is_c_fulled_used": C_fulled_used,
        # C的最大使用量
        "c_max_usage": C_max_usage
    }
    '''
    
    # 第二次资源分配
    second_analyse_result = analyser.analyse_resources_second_time(
        project_part,
        lut,
        ff,
        bram,
        dsp,
        bram_threshold,
        lut_threshold,
        try_increase_c_bandwidth,
        optimize,
        im2col_shape,
        calculation_graph,
        first_analyse_result,
        first_tensor_expression
    )
    '''
    second_analyse_result = {
        # bram组数(组边长小于512时，此值为0)
        "bram_group": bram_group,   
        # 每组bram中C的列数
        "bram_col_c_need_per_bram_group": bram_col_C_need_per_bram_group, 
        # C需要的bram深度  
        "depth_c_need_per_bram_col": depth_C_need_per_bram_col,     
        # 所需要的总的bram数
        "total_bram_need": total_bram_need,     
        # 可用的bram数
        "bram_avaliable": int(bram_threshold*bram),
        # 支持的最大矩阵(也即bram组的边长)
        "max_matrix_len_support": max_len_support,  
        # 支持的最小矩阵
        "min_matrix_len_support": min_len_support,  
        # 每组bram分配的计算单元组数
        "calc_unit_per_bram_group": calc_uint_per_bram_group,   
        # 需要的总lut数
        "total_lut_need": total_lut_need,           
        # 可用的lut数
        "lut_avaliable": int(lut_threshold*lut)
        # 是否采用了更激进的分配
        "more_radical_allocation": more_radical_allocation   
    }
    '''

    # 第二次拆分张量表达式
    sencond_tensor_expression = analyser.split_tensor_expression_second_time(
        project_part,
        lut,
        ff,
        bram,
        dsp,
        bram_threshold,
        lut_threshold,
        try_increase_c_bandwidth,
        optimize,
        first_analyse_result,
        first_tensor_expression,
        second_analyse_result,
        im2col_shape,
        calculation_graph
    )

    exit()








    code = conv.gen_conv(
        MODULE_NAME="conv",
        MUX_WIDTH=2,
        DATA_WIDTH=8,
        DATA_NUMBER=256,
        OUTPUT_PORTS=[16, 32, 64, 256],
        ZERO_X=[0, 34, 84],
        ZERO_W=[78, 129, 132],
        DEBUG=True,
    )
    with open("output/conv.v", "w") as f:
        f.write(code)

    code = fc.gen_fc(
        MODULE_NAME="fc",
        MUX_WIDTH=1,
        DATA_WIDTH=8,
        DATA_NUMBER=8,
        HIDDEN_LEN=3136,
        OUTPUT_LEN=10,
        BIAS=[(np.random.rand(10)*100).astype("int32")],
        COE=[0.657257],
        RSHIFT=[9],
        ZERO_X=[131],
        ZERO_W=[121],
        ZERO_Y=[106],
        QMAX=255,
        DEBUG=True,
    )
    with open("output/fc.v", "w") as f:
        f.write(code)

    code = post_process.post_process(
        MODULE_NAME="post_process",
        MUX_WIDTH=2,
        DATA_WIDTH=32,
        DATA_NUMBER=16,
        OUT_DATA_WIDTH=8,
        BIAS=[
            np.random.randint(-32768, 32768, 16).astype("int32"),
            np.random.randint(-32768, 32768, 32).astype("int32"),
            np.random.randint(-32768, 32768, 64).astype("int32"),
        ],
        COE=[0.247298, 0.711928, 0.818192],
        RSHIFT=[9, 9, 9],
        ZERO_Y=[112, 108, 129],
        QMAX=255,
        DEBUG=True                          # debug
    )
    with open("output/post_process.v", "w") as f:
        f.write(code)