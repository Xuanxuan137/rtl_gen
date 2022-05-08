
import math

import numpy as np
import op
from util import *


def set_resources(
    project_part,
    lut,
    ff,
    bram,
    dsp
):
    '''
    对于未给定的资源数量，根据芯片型号自动设置资源数量
    '''
    xxlog("Setting resources...")
    if(lut is not None and
       ff is not None and
       bram is not None and
       dsp is not None):
        xxlog("Set lut:%d, ff:%d, bram:%d, dsp:%d"%(lut, ff, bram, dsp))
        return lut, ff, bram, dsp

    # 根据芯片设定预知的资源
    if(project_part == "xc7z020clg400-1" or 
       project_part == "xc7z020clg400-2" or
       project_part == "xc7z020clg400-3"):
        known_lut = 53200
        known_ff = 106400
        known_bram = 140
        known_dsp = 220
    
    # 如果某个资源没有给出，将其设为预知资源数量
    if(lut is None):
        lut = known_lut
    if(ff is None):
        ff = known_ff
    if(bram is None):
        bram = known_bram
    if(dsp is None):
        dsp = known_dsp
    
    xxlog("Set lut:%d, ff:%d, bram:%d, dsp:%d"%(lut, ff, bram, dsp))
    return lut, ff, bram, dsp


def infer_im2col_shape(calculation_graph):
    '''
    推算计算图中卷积算子im2col后的矩阵边长
    '''
    xxlog("Inferring matrix shape after im2col...")
    im2col_shape = []
    for node in calculation_graph:
        if(type(node) != op.QConv2d):
            xxlog("Detected op %s which is not QConv2d. Pass"%(type(node)))
            continue
        output_channel = node.output_channel
        input_channel = node.input_channel
        kernel_size = node.kernel_size
        stride = node.stride
        padding = node.padding
        dilation = node.dilation
        output_shape = node.output_shape
        weight_shape = [output_channel, input_channel, kernel_size[0], kernel_size[1]]
        bias_shape = [output_channel]
        
        weight_matrix_row = weight_shape[0]
        weight_matrix_col = weight_shape[1] * weight_shape[2] * weight_shape[3]
        
        feature_map_matrix_row = weight_shape[1] * weight_shape[2] * weight_shape[3]
        feature_map_matrix_col = output_shape[2] * output_shape[3]
        
        matrix_shape_after_im2col = ([weight_matrix_row, weight_matrix_col], 
            [feature_map_matrix_row, feature_map_matrix_col])
        im2col_shape.append(matrix_shape_after_im2col)
        xxlog("Detected QConv2d with shape after im2col: %s x %s. Min side length: %s"%(
            matrix_shape_after_im2col[0], matrix_shape_after_im2col[1], min(matrix_shape_after_im2col[0][0],
            matrix_shape_after_im2col[0][1], matrix_shape_after_im2col[1][0], matrix_shape_after_im2col[1][1])
        ))
    xxlog("Infer matrix shape after im2col finished")

    return im2col_shape


def analyse_resources_first_time(
    project_part,
    lut,
    ff,
    bram,
    dsp,
    bram_threshold,
    lut_threshold,
    im2col_shape
):
    '''
    第一次分析资源
    0. 保证bram带宽为2的幂
    1. 按照A:B:C=1:1:4分配bram
    2. 由于bram深度为512，所以如果A和B的深度超过了512，就浪费了带宽，所以初始时限制A和B的最大深度为512，即单组bram最大支持512x512的矩阵乘法。
    3. 对于bram只能容纳小于等于一组的，直接计算即可
    4. 对于bram能够容纳大于等于1组的，只要完整组。
    5. 保证组的数量为2的幂
    6. 对于bram能够容纳大于1组的，如果资源足够使组的边长和深度均翻倍，则增加单个组的大小。即如果能够容纳4个512组，则使组大小变为1024。
    如果不足以使组大小增加，则将这两组横向排列以增加带宽。
    显然，这时可以发现，组的数量只能为1和2，当达到4的时候，组就可以合并，变成更大的单个组。
    7. 计算给当前每组bram分配一组计算单元时，lut的使用量。
    8. 如果lut充足，尝试给每组bram分配的计算单元翻倍。
    9. 保证每组bram分配的计算单元组数为2的幂
    '''
    xxlog("Analyse resources first time")

    space_per_bram = 4608   # 每块bram的空间4608字节
    max_len_per_group = 512     # 一组bram支持的最大矩阵边长
    
    matrix_len = 8  # 支持的最大矩阵边长。不能小于8，因为一块bram就能支持8
    bram_group = 0  # 完整的512 bram组数
    while(True):
        # 计算当前matrix_len需要的空间
        space_A_need = matrix_len * matrix_len
        space_B_need = matrix_len * matrix_len
        bram_A_need = matrix_len // 8
        bram_B_need = matrix_len // 8
        space_C_need = matrix_len * matrix_len * 4
        bram_C_need = math.ceil(space_C_need / space_per_bram)
        total_bram_need = bram_A_need + bram_B_need + bram_C_need
        if(total_bram_need <= bram_threshold * bram):
            xxlog("Group: %d. Matrix len: %d. Bram for A, B, C: %d, %d, %d. " \
                "Bram current group: %d. Bram total: %d. Avaliable: %d."%(
                bram_group, matrix_len, bram_A_need, bram_B_need, bram_C_need, 
                total_bram_need, total_bram_need,
                bram_threshold*bram
            ))
            if(matrix_len == max_len_per_group):
                # 如果一组bram满512了，此时应该新开一组bram
                bram_group += 1
                matrix_len = 0
                xxlog("Complete group number: %d."%(bram_group))
                break
            else:
                matrix_len *= 2
        else:
            # 此时bram已经用超了，应该回退一步
            if(matrix_len == 8):
                matrix_len = 0
            else:
                matrix_len //= 2
            bram_A_need = matrix_len // 8
            bram_B_need = matrix_len // 8
            space_C_need = matrix_len * matrix_len * 4
            bram_C_need = math.ceil(space_C_need / space_per_bram)
            total_bram_need = bram_A_need + bram_B_need + bram_C_need
            break
    
    if(bram_group >= 1):
        # 说明有完整的组，此时应该尝试给组翻倍
        xxlog("Found a complete group in first analyse: Bram usage: %d, " \
            "bram avaliable: %d. Try to double the bram"%(total_bram_need,
            bram_threshold*bram))
        while(True):
            bram_group *= 2
            total_bram_need *= 2
            if(total_bram_need <= bram_threshold*bram):
                xxlog("Group: %d, bram total: %d, bram avaliable: %d"%(
                    bram_group, total_bram_need, bram_threshold*bram
                ))
            else:
                # 此时bram已经用超了
                bram_group //= 2
                total_bram_need //= 2
                xxlog("First analyse result: Group: %d, bram usage: %d, " \
                    "bram avaliable: %d"%(bram_group, total_bram_need, 
                    bram_threshold*bram))
                break
    else:
        # 没有完整的组，分配结束
        xxlog("First analyse result: No complete group. Matrix_len: %d, " \
            "bram usage: %d, bram avaliable: %d"%(matrix_len, 
            total_bram_need, bram_threshold*bram))
    
    '''
    检查计算资源是否足够
    '''
    lut_need_per_mult = 61
    lut_need_per_add = 8
    lut_need_per_sub = 8
    total_mult = bram_group * max_len_per_group + matrix_len
    total_add = bram_group * (max_len_per_group-1) + matrix_len-1
    total_sub = bram_group * max_len_per_group * 2 + matrix_len * 2
    total_lut_need = (total_mult * lut_need_per_mult + 
        total_add * lut_need_per_add + total_sub * lut_need_per_sub)
    xxlog("Lut need(no consider dsp): %d. Lut avaliable: %d"%(
        total_lut_need, lut_threshold*lut))
    
    calc_uint_per_bram_group = 1
    if(total_lut_need <= lut_threshold*lut):
        xxlog("Try to double calculation unit")
        while(True):
            total_lut_need *= 2
            calc_uint_per_bram_group *= 2
            if(total_lut_need <= lut_threshold*lut):
                xxlog("Calculation unit per bram group: %d, " \
                    "Total lut need: %d. Lut avaliable: %d"%(
                        calc_uint_per_bram_group, total_lut_need,
                        lut_threshold*lut
                    ))
            else:
                # 此时已经用超了
                total_lut_need //= 2
                calc_uint_per_bram_group //= 2
                xxlog("First lut allocate finished: Calculation uint " \
                    "per group: %d. Total lut need: %d. Lut avaliable: %d"%(
                        calc_uint_per_bram_group, total_lut_need, 
                        lut_threshold*lut))
                break
    
    # 此时初次分配完成
    xxlog("First bram analyse and lut allocate finished, the result is " \
        "shown below:\n" \
        "\tComplete bram group: %d\n" \
        "\tIncomplete bram group len: %d\n" \
        "\tTotal bram need: %d\n" \
        "\tBram avaliable: %d\n" \
        "\tCalculation unit per bram group: %d\n" \
        "\tTotal lut need: %d\n" \
        "\tLut avaliable: %d"%(
            bram_group, matrix_len, total_bram_need, bram_threshold*bram,
            calc_uint_per_bram_group, total_lut_need, lut_threshold*lut
        ))


    '''
    修正C占用空间
    -- 拆分张量表达式
    对矩阵进行切块
    1. 尽量切大块，但不超过片上支持的最大边长
    2. 矩阵的相乘边要为2的幂
    3. 矩阵的结果边要为合适的长度，使得本矩阵块能够填满bram的一行
    返回切块结果
    '''