

import math
from multiprocessing.sharedctypes import Value

import numpy as np
import op
from util import *


def get_bram_usage(width, depth):
    '''
    根据bram位宽和深度获取bram使用量
    由于不知道xilinx是怎么计算bram用量的，这一步目前只能查表
    '''
    if(width == 64):
        usage = {
            # depth: (36K usage, 18K usage)
            16: (1, 0),
            32: (1, 0),
            64: (1, 0),
            128: (1, 0),
            256: (1, 0),
            512: (1, 0),
            1024: (2, 0), 
            1536: (3, 0),
            2048: (4, 0),
            3072: (6, 0),
            4096: (7, 1),
            5120: (8, 3),
            6144: (10, 2),
            7168: (11, 4),
            8192: (14, 1),
            10240: (17, 2),
            12288: (21, 1),
            14336: (24, 2),
            16384: (28, 1),
            24576: (43, 0),
            32768: (57, 0),
            49152: (85, 1),
            65536: (114, 0),
        }
        try: 
            return usage[depth]
        except:
            xxlog("Width=%d, Depth=%d not supported yet"%(width, depth))
            raise ValueError("Width=%d, Depth=%d not supported yet"%(width, depth))
    else:
        xxlog("Width=%d not supported yet"%(width), XXError())
        raise ValueError("Width=%d not supported yet"%(width))


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


def search_conv2d(calculation_graph, n):
    '''
    从calculation_graph中找到第n个conv2d并返回
    '''
    count = 0
    for node in calculation_graph:
        if(type(node) == op.QConv2d):
            count += 1
            if(count == n):
                return node
    return None


def is_power_of_2(value):
    '''
    判断是否是2的幂
    '''
    index = 0
    while(index <= 31):
        if(value == 2**index):
            return True
        index += 1
    return False


def fit_to_power_of_2(value, min_len_support):
    '''
    value: 输入的矩阵边长
    min_len_support: 片上的最小矩阵边长
    用于在切分矩阵时将矩阵边长适应到2的幂
    方法：
    1. 如果已经是2的幂，且不小于min_len_support，直接返回
    2. 如果小于min_len_support，则向上扩展到min_len_support
    3. 否则，如果小于64，则向上扩展到最近的2的幂
    4. 否则，如果大于下一级2的幂与上一级2的幂的中点，则向上扩展，否则向下压缩
    '''
    if(is_power_of_2(value) and value>=min_len_support):
        return value
    if(value < min_len_support):
        return min_len_support
    if(value < 64):
        n = 0
        while(2**n < value):
            n += 1
        return 2**n
    n = 0
    while(2**n < value):
        n += 1
    upper_bound = 2**n
    lower_bound = 2**(n-1)
    if(value > (upper_bound + lower_bound) // 2):
        return upper_bound
    return lower_bound


def analyse_resources_first_time(
    project_part,
    lut,
    ff,
    bram,
    dsp,
    bram_threshold,
    lut_threshold,
    im2col_shape,
    calculation_graph
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
    width_per_bram = 64         # 使用的一块bram的宽度
    depth_per_bram = 512        # 一块bram的深度
    
    matrix_len = 8  # 支持的最大矩阵边长。不能小于8，因为一块bram就能支持8
    bram_group = 0  # 完整的512 bram组数
    while(True):
        # 计算当前matrix_len需要的空间
        space_A_need = matrix_len * matrix_len
        space_B_need = matrix_len * matrix_len
        bram_A_need = matrix_len // 8
        bram_B_need = matrix_len // 8
        space_C_need = matrix_len * matrix_len * 4
        bram_C_need = int(math.ceil(space_C_need / space_per_bram))
        total_bram_need = bram_A_need + bram_B_need + bram_C_need
        if(total_bram_need <= int(bram_threshold*bram)):
            xxlog("Group: %d. Matrix len: %d. Bram for A, B, C: %d, %d, %d. " \
                "Bram current group: %d. Bram total: %d. Avaliable: %d."%(
                bram_group, matrix_len, bram_A_need, bram_B_need, bram_C_need, 
                total_bram_need, total_bram_need,
                int(bram_threshold*bram)
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
            bram_C_need = int(math.ceil(space_C_need / space_per_bram))
            total_bram_need = bram_A_need + bram_B_need + bram_C_need
            break
    
    if(bram_group >= 1):
        # 说明有完整的组，此时应该尝试给组翻倍
        xxlog("Found a complete group in first analyse: Bram usage: %d, " \
            "bram avaliable: %d. Try to double the bram"%(total_bram_need,
            int(bram_threshold*bram)))
        while(True):
            bram_group *= 2
            total_bram_need *= 2
            if(total_bram_need <= int(bram_threshold*bram)):
                xxlog("Group: %d, bram total: %d, bram avaliable: %d"%(
                    bram_group, total_bram_need, int(bram_threshold*bram)
                ))
            else:
                # 此时bram已经用超了
                bram_group //= 2
                total_bram_need //= 2
                xxlog("First analyse result: Group: %d, bram usage: %d, " \
                    "bram avaliable: %d"%(bram_group, total_bram_need, 
                    int(bram_threshold*bram)))
                break
    else:
        # 没有完整的组，分配结束
        xxlog("First analyse result: No complete group. Matrix_len: %d, " \
            "bram usage: %d, bram avaliable: %d"%(matrix_len, 
            total_bram_need, int(bram_threshold*bram)))
    
    '''
    检查计算资源是否足够
    '''
    lut_need_per_mult = 61
    lut_need_per_add = 8
    lut_need_per_sub = 8
    lut_counter_per_dsp = 25    # 每个dsp能够抵消的lut数量(估计值, 不一定准确)
    total_mult = bram_group * max_len_per_group + matrix_len
    total_add = bram_group * (max_len_per_group-1) + matrix_len-1
    total_sub = bram_group * max_len_per_group * 2 + matrix_len * 2
    total_lut_need = (total_mult * lut_need_per_mult + 
        total_add * lut_need_per_add + total_sub * lut_need_per_sub)
    xxlog("Lut need(no consider dsp): %d. Lut avaliable: %d"%(
        total_lut_need, int(lut_threshold*lut)))
    
    calc_uint_per_bram_group = 1
    if(total_lut_need - lut_counter_per_dsp*dsp <= int(lut_threshold*lut)):
        # 如果资源充足，增加每个bram组的计算单元数量
        xxlog("Try to double calculation unit")
        while(True):
            total_lut_need *= 2
            calc_uint_per_bram_group *= 2
            if(total_lut_need - lut_counter_per_dsp*dsp <= int(lut_threshold*lut)):
                xxlog("Calculation unit per bram group: %d, " \
                    "Total lut need: %d. Lut avaliable: %d"%(
                        calc_uint_per_bram_group, total_lut_need,
                        int(lut_threshold*lut)
                    ))
            else:
                # 此时已经用超了
                total_lut_need //= 2
                calc_uint_per_bram_group //= 2
                xxlog("First lut allocate finished: Calculation uint " \
                    "per group: %d. Total lut need: %d. Lut avaliable: %d"%(
                        calc_uint_per_bram_group, total_lut_need, 
                        int(lut_threshold*lut)))
                break
    else:
        # 如果资源不够，进行缩减bram数量
        xxlog("Lut not enough, try to decrease...")
        solved = False
        # 尝试减少完整的bram组
        while(bram_group > 1):
            bram_group //= 2
            total_lut_need //= 2
            xxlog("Decrease bram group to %d"%(bram_group))
            if(total_lut_need - lut_counter_per_dsp*dsp <= int(lut_threshold*lut)):
                solved = True
                xxlog("Lut enough now, need:%d, avaliable:%d"%(
                    total_lut_need, int(lut_threshold*lut)))
                break
        # 如果bram组不多余1组，尝试缩减最大矩阵边长
        if(not solved):
            if(bram_group >= 1):
                matrix_len = 512
                bram_group = 0
            while(matrix_len >= 8):
                matrix_len //= 2
                total_lut_need //= 2
                xxlog("Decrease matrix_len to %d"%(matrix_len))
                if(total_lut_need - lut_counter_per_dsp*dsp <= int(lut_threshold*lut)):
                    solved = True
                    xxlog("Lut enough now, need:%d, avaliable:%d"%(
                        total_lut_need, int(lut_threshold*lut)))
                    break
        # 如果矩阵边长小于8，lut仍不够，报错
        if(not solved):
            xxlog("Device is too small to accelerate neural network", XXError())
            raise ValueError("Device is too small to accelerate neural network")
        # 解决后，重新计算bram需求
        bram_A_need = matrix_len // 8
        bram_B_need = matrix_len // 8
        space_C_need = matrix_len * matrix_len * 4
        bram_C_need = int(math.ceil(space_C_need / space_per_bram))
        total_bram_need = bram_A_need + bram_B_need + bram_C_need

    
    # 片上支持的最大矩阵
    max_len_support = max_len_per_group if(bram_group >= 1) else matrix_len
    temp_bram_group = bram_group
    while(True):
        # bram_group每4倍，最大矩阵边长2倍
        temp_bram_group //= 4
        if(temp_bram_group >= 1):
            max_len_support *= 2
        else:
            break
    # 片上支持的最小矩阵(保证最小矩阵至少能填满bram组的一行)
    min_len_support = 8
    while(min_len_support**2 < max_len_support):
        min_len_support *= 2

    # 此时初次分配完成
    xxlog("First bram analyse and lut allocate finished, the result is " \
        "shown below:\n" \
        "\tComplete bram group: %d\n" \
        "\tIncomplete bram group len: %d\n" \
        "\tTotal bram need: %d\n" \
        "\tBram avaliable: %d\n" \
        "\tMax matrix len support: %d\n" \
        "\tMin matrix len support: %d\n" \
        "\tCalculation unit per bram group: %d\n" \
        "\tTotal lut need: %d\n" \
        "\tLut avaliable: %d"%(
            bram_group, matrix_len, total_bram_need, int(bram_threshold*bram),
            max_len_support, min_len_support, calc_uint_per_bram_group, 
            total_lut_need, int(lut_threshold*lut)
        ))
    


    '''
    修正C占用空间
    -- 拆分张量表达式
    对矩阵进行切块
    1. 尽量切大块，但不超过片上支持的最大边长
    2. 矩阵的相乘边要为2的幂
    3. 矩阵的结果边要为合适的长度，使得本矩阵块能够填满bram的一行
    得到切块结果
    '''
    while(True):
        original_shape = []
        xxlog("Copy im2col_shape to original_shape...")
        for shape in im2col_shape:
            original_shape.append(([shape[0][0], shape[0][1]], 
                [shape[1][0], shape[1][1]]))
        divided_border = []     # 切分结果
        xxlog("Divide im2col matrix...")
        for n, shape in enumerate(original_shape):
            xxlog("Dividing conv2d layer %d: %s"%(
                n, search_conv2d(calculation_graph, n+1)))
            xxlog("Currect layer shape: %s, %s"%(shape[0], shape[1]))
            height_A = shape[0][0]
            width_A = shape[0][1]
            height_B = shape[1][0]
            width_B = shape[1][1]
            start_h_A = 0
            start_w_A = 0
            start_h_B = 0
            start_w_B = 0
            current_layer_divided_border_A = []
            current_layer_divided_border_B = []
            # A和b是分别切分的。除了相乘边需要保持一致外，其他边不需要一致。
            # 先切分A
            while(True):
                if(height_A - start_h_A >= max_len_support and
                   width_A - start_w_A >= max_len_support):
                    # 如果足够切出来最大的块
                    cut_height = max_len_support
                    cut_width = max_len_support
                else:
                    # 如果不足够切出来最大的块
                    if(height_A - start_h_A < max_len_support and
                       width_A - start_w_A < max_len_support):
                        # 如果height和width都不够
                        cut_height = fit_to_power_of_2(
                            height_A - start_h_A, min_len_support)
                        cut_width = fit_to_power_of_2(
                            width_A - start_w_A, min_len_support)
                    elif(height_A - start_h_A < max_len_support):
                        # 如果height不够
                        cut_height = fit_to_power_of_2(
                            height_A - start_h_A, min_len_support)
                        cut_width = max_len_support
                    elif(width_A - start_w_A < max_len_support):
                        # 如果width不够
                        cut_height = max_len_support
                        cut_width = fit_to_power_of_2(
                            width_A - start_w_A, min_len_support)
                    else:
                        xxlog("Error when dividing matrix: height_A:%d, " \
                            "start_h_A:%d, width_A:%d, start_w_A:%d, " \
                            "max_len_support:%d"%(height_A, start_h_A, width_A, 
                            start_w_A, max_len_support), XXError())
                        raise ValueError("Error when dividing matrix")
                # 保存切分结果
                current_layer_divided_border_A.append(
                    [start_h_A, start_h_A+cut_height, start_w_A, start_w_A+cut_width]
                )
                xxlog("Cut [%d:%d, %d:%d] from A"%(
                    current_layer_divided_border_A[-1][0], 
                    current_layer_divided_border_A[-1][1],
                    current_layer_divided_border_A[-1][2],
                    current_layer_divided_border_A[-1][3]))
                # 修改下一次切分起始点
                if(start_w_A + cut_width >= width_A):
                    # width方向达到最大值，需要换行
                    if(start_h_A + cut_height >= height_A):
                        # 如果height方向也达到最大值，结束
                        break
                    else:
                        start_h_A += cut_height
                        start_w_A = 0
                else:
                    start_w_A += cut_width
            # 再切分B
            while(True):
                if(height_B - start_h_B >= max_len_support and
                   width_B - start_w_B >= max_len_support):
                    # 如果足够切出来最大的块
                    cut_height = max_len_support
                    cut_width = max_len_support
                else:
                    # 如果不足够切出来最大的块
                    if(height_B - start_h_B < max_len_support and
                       width_B - start_w_B < max_len_support):
                        # 如果height和width都不够
                        cut_height = fit_to_power_of_2(
                            height_B - start_h_B, min_len_support)
                        cut_width = fit_to_power_of_2(
                            width_B - start_w_B, min_len_support)
                    elif(height_B - start_h_B < max_len_support):
                        # 如果height不够
                        cut_height = fit_to_power_of_2(
                            height_B - start_h_B, min_len_support)
                        cut_width = max_len_support
                    elif(width_B - start_w_B < max_len_support):
                        # 如果width不够
                        cut_height = max_len_support
                        cut_width = fit_to_power_of_2(
                            width_B - start_w_B, min_len_support)
                    else:
                        xxlog("Error when dividing matrix: height_B:%d, " \
                            "start_h_B:%d, width_B:%d, start_w_B:%d, " \
                            "max_len_support:%d"%(height_B, start_h_B, width_B, 
                            start_w_B, max_len_support), XXError())
                        raise ValueError("Error when dividing matrix")
                # 保存切分结果
                current_layer_divided_border_B.append(
                    [start_h_B, start_h_B+cut_height, start_w_B, start_w_B+cut_width]
                )
                xxlog("Cut [%d:%d, %d:%d] from B"%(
                    current_layer_divided_border_B[-1][0], 
                    current_layer_divided_border_B[-1][1],
                    current_layer_divided_border_B[-1][2],
                    current_layer_divided_border_B[-1][3]))
                # 修改下一次切分起始点
                if(start_w_B + cut_width >= width_B):
                    # width方向达到最大值，需要换行
                    if(start_h_B + cut_height >= height_B):
                        # 如果height方向也达到最大值，结束
                        break
                    else:
                        start_h_B += cut_height
                        start_w_B = 0
                else:
                    start_w_B += cut_width
            divided_border.append((current_layer_divided_border_A,
                current_layer_divided_border_B))
        xxlog("Divide im2col matrix finished")
        
        '''
        校验切分结果：
        1. A的上边应该与B的左边切分结果相同
        2. A的不同行的上边切分结果应相同(不校验了，懒)
        3. B的不同列的左边的切分结果应相同(不校验了，懒)
        '''
        xxlog("Checking divided result...")
        for layer in divided_border:
            border_A = layer[0]
            border_B = layer[1]
            cut_result_A = []
            cut_result_B = []
            last_start_h_B = -1
            for border in border_A:
                if(border[0] == 0):
                    # 遍历第一行，记录每一块的width
                    cut_result_A.append(border[3] - border[2])
            for border in border_B:
                if(border[0] != last_start_h_B):
                    cut_result_B.append(border[1] - border[0])
                    last_start_h_B = border[0]
            if(cut_result_A != cut_result_B):
                xxlog("Check divide result failed", XXError())
                raise ValueError("Check divide result failed")
        xxlog("Check divide result passed")
    
                    
        '''
        修正C占用空间
        -- 重新计算C需要的空间
        1. 越小的矩阵块在计算时，每周期得到的结果数越多，这决定了C需要的带宽。
        2. C所需要的bram数需要在1的基础上，再乘上每块里bram分配的计算单元组数。
        3. 根据C需要的带宽计算C实际需要的bram数
        4. 如果片上资源能够容纳修正后的ABC，则完成
        5. 如果不能容纳，则减少ABC的bram分配，然后回到上一步重新修正
        ''' 
        xxlog("Fixing bram_C_need...")
        # 1. 从切块结果中找到相乘边(A的上边和B的左边)最小的矩阵块(只需要找A的上边即可)
        xxlog("Finding min matrix block need to calculate...")
        min_matrix_len = 2147483647
        for layer in divided_border:
            border_A = layer[0]
            border_B = layer[1]
            cut_result_A = []
            for border in border_A:
                if(border[0] == 0):
                    cut_result_A.append(border[3] - border[2])
            current_layer_min_matrix_len = min(cut_result_A)
            min_matrix_len = min(min_matrix_len, current_layer_min_matrix_len)
        xxlog("Min matrix block need to calculate is %d"%(min_matrix_len))
        # 每周期每组计算单元得到的结果数
        result_per_cycle_per_calc_unit = max_len_support // min_matrix_len 
        xxlog("Result get per cycle per calculation unit: %d"%(
            result_per_cycle_per_calc_unit))
        # 2. C所需要的bram数需要在1的基础上，再乘上每块里bram分配的计算单元组数。
        # 每周期得到的结果数
        result_per_cycle = result_per_cycle_per_calc_unit * calc_uint_per_bram_group
        xxlog("Result per cycle: %d"%(result_per_cycle))
        # 3. 根据C需要的带宽计算C实际需要的bram数
        result_per_bram = 2 # 结果为32bit的条件下，每列bram每周期能写入的结果数
        # C需要的bram列数
        bram_col_C_need = result_per_cycle / result_per_bram 
        xxlog("Bram col C need: %d"%(bram_col_C_need))
        # 每个bram_group C需要的空间
        if(bram_group >= 1):
            space_C_need_per_bram_group = 512 * 512 * 4
            space_C_need = space_C_need_per_bram_group * bram_group
        else:
            space_C_need = max_len_support * max_len_support * 4
        xxlog("Space C need: %d"%(space_C_need))
        # C需要每列bram的空间
        space_C_need_per_bram_col = space_C_need / bram_col_C_need
        xxlog("Space C need per bram col: %d"%(space_C_need_per_bram_col))
        # C需要每列bram的深度
        depth_C_need_per_bram_col = space_C_need_per_bram_col / 8
        xxlog("Depth C need per bram col: %d"%(depth_C_need_per_bram_col))
        # C需要每列bram的个数(由于不知道xilinx bram的计算方式，这一步目前只能查表)
        bram36_C_need_per_col = get_bram_usage(width_per_bram, 
            depth_C_need_per_bram_col)[0]
        bram18_C_need_per_col = get_bram_usage(width_per_bram, 
            depth_C_need_per_bram_col)[1]
        xxlog("Bram36 C need per col: %d, bram18 C need per col: %d"%(
            bram36_C_need_per_col, bram18_C_need_per_col))
        # C需要的bram个数
        bram_C_need = (bram_col_C_need * bram36_C_need_per_col + 
            int(math.ceil(bram_col_C_need * bram18_C_need_per_col / 2)))
        xxlog("Bram C need: %d"%(bram_C_need))
        # 总需要的bram个数
        if(bram_group >= 1):
            bram_A_need = bram_group * 64
            bram_B_need = bram_group * 64
        else:
            bram_A_need = max_len_support // 8
            bram_B_need = max_len_support // 8
        xxlog("Bram A need: %d, Bram B need: %d"%(bram_A_need, bram_B_need))
        total_bram_need = bram_A_need + bram_B_need + bram_C_need
        xxlog("Total bram need: %d"%(total_bram_need))
        
        # 如果C修正后bram仍然足够，则可以结束
        if(total_bram_need <= int(bram_threshold*bram)):
            xxlog("Bram enough after fixed C")
            break
        
        # C修正后，bram不足，对bram使用进行修正
        xxlog("Bram not enough after fixed C, try to decrease bram...")
        if(bram_group > 1):
            bram_group //= 2
        elif(bram_group == 1):
            bram_group = 0
            matrix_len = 512
            matrix_len //= 2
        else:
            matrix_len //= 2
        if(matrix_len < 8):
            xxlog("Device is too small to accelerate neural network", XXError())
            raise ValueError("Device is too small to accelerate neural network")
        bram_A_need = matrix_len // 8
        bram_B_need = matrix_len // 8
        space_C_need = matrix_len * matrix_len * 4
        bram_C_need = int(math.ceil(space_C_need / space_per_bram))
        total_bram_need = bram_A_need + bram_B_need + bram_C_need
        max_len_support = max_len_per_group if(bram_group >= 1) else matrix_len
        min_len_support = 8
        while(min_len_support**2 < max_len_support):
            min_len_support *= 2
        # 同时对lut使用进行修正
        xxlog("Divide bram usage by 2. Try to fix lut usage...")
        total_mult = bram_group * max_len_per_group + matrix_len
        total_add = bram_group * (max_len_per_group-1) + matrix_len-1
        total_sub = bram_group * max_len_per_group * 2 + matrix_len * 2
        total_lut_need = (total_mult * lut_need_per_mult + 
            total_add * lut_need_per_add + total_sub * lut_need_per_sub)
        xxlog("Lut need(no consider dsp): %d. Lut avaliable: %d"%(
            total_lut_need, int(lut_threshold*lut)))
        
        calc_uint_per_bram_group = 1
        if(total_lut_need - lut_counter_per_dsp*dsp <= int(lut_threshold*lut)):
            # 如果资源充足，增加每个bram组的计算单元数量
            xxlog("Try to double calculation unit")
            while(True):
                total_lut_need *= 2
                calc_uint_per_bram_group *= 2
                if(total_lut_need - lut_counter_per_dsp*dsp <= int(lut_threshold*lut)):
                    xxlog("Calculation unit per bram group: %d, " \
                        "Total lut need: %d. Lut avaliable: %d"%(
                            calc_uint_per_bram_group, total_lut_need,
                            int(lut_threshold*lut)
                        ))
                else:
                    # 此时已经用超了
                    total_lut_need //= 2
                    calc_uint_per_bram_group //= 2
                    xxlog("First lut allocate finished: Calculation uint " \
                        "per group: %d. Total lut need: %d. Lut avaliable: %d"%(
                            calc_uint_per_bram_group, total_lut_need, 
                            int(lut_threshold*lut)))
                    break
        xxlog("Bram decrease and lut allocate finished, the result is " \
            "shown below:\n" \
            "\tComplete bram group: %d\n" \
            "\tIncomplete bram group len: %d\n" \
            "\tTotal bram need: %d\n" \
            "\tBram avaliable: %d\n" \
            "\tMax matrix len support: %d\n" \
            "\tMin matrix len support: %d\n" \
            "\tCalculation unit per bram group: %d\n" \
            "\tTotal lut need: %d\n" \
            "\tLut avaliable: %d"%(
                bram_group, matrix_len, total_bram_need, int(bram_threshold*bram),
                max_len_support, min_len_support, calc_uint_per_bram_group, 
                total_lut_need, int(lut_threshold*lut)
            ))
        xxlog("Check again...")

    # 返回结果
    xxlog("Bram decrease and lut allocate finished, the result is " \
            "shown below:\n" \
            "\tComplete bram group: %d\n" \
            "\tIncomplete bram group len: %d\n" \
            "\tTotal bram need: %d\n" \
            "\tBram avaliable: %d\n" \
            "\tMax matrix len support: %d\n" \
            "\tMin matrix len support: %d\n" \
            "\tCalculation unit per bram group: %d\n" \
            "\tTotal lut need: %d\n" \
            "\tLut avaliable: %d"%(
                bram_group, matrix_len, total_bram_need, int(bram_threshold*bram),
                max_len_support, min_len_support, calc_uint_per_bram_group, 
                total_lut_need, int(lut_threshold*lut)))
    return {
        "bram_group": bram_group,
        "matrix_len": matrix_len,
        "total_bram_need": total_bram_need,
        "bram_avaliable": int(bram_threshold*bram),
        "max_matrix_len_support": max_len_support,
        "min_matrix_len_support": min_len_support,
        "calc_unit_per_bram_group": calc_uint_per_bram_group,
        "total_lut_need": total_lut_need,
        "lut_avaliable": int(lut_threshold*lut)
    }