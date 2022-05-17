
import math

import numpy as np
import op
from util import *


def is_large_matrix(matrix_space):
    if(matrix_space >= 16384):
        return True
    return False


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
            2560: (5, 0),
            3072: (6, 0),
            3584: (7, 0),
            4096: (7, 1),
            4608: (9, 0),
            5120: (8, 3),
            5632: (8, 5),
            6144: (10, 2),
            6656: (8, 9),
            7168: (11, 4),
            7680: (10, 8),
            8192: (14, 1),
            8704: (14, 4),
            9216: (15, 3),
            9728: (15, 5),
            10240: (17, 2),
            10752: (15, 9),
            11264: (18, 4),
            11776: (17, 8),
            12288: (21, 1),
            12800: (21, 4),
            13312: (22, 3),
            13824: (22, 5),
            14336: (24, 2),
            14848: (22, 9),
            15360: (25, 4),
            15872: (24, 8),
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
            raise ValueError("Width=%d, Depth=%d not supported yet"%(
                width, depth))
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
        weight_shape = [output_channel, input_channel, 
            kernel_size[0], kernel_size[1]]
        bias_shape = [output_channel]
        
        weight_matrix_row = weight_shape[0]
        weight_matrix_col = weight_shape[1] * weight_shape[2] * weight_shape[3]
        
        feature_map_matrix_row = weight_shape[1] * weight_shape[2] \
            * weight_shape[3]
        feature_map_matrix_col = output_shape[2] * output_shape[3]
        
        matrix_shape_after_im2col = ([weight_matrix_row, weight_matrix_col], 
            [feature_map_matrix_row, feature_map_matrix_col])
        im2col_shape.append(matrix_shape_after_im2col)
        xxlog("Detected QConv2d with shape after im2col: %s x %s. " \
            "Min side length: %s"%(
            matrix_shape_after_im2col[0], matrix_shape_after_im2col[1], 
            min(matrix_shape_after_im2col[0][0],
            matrix_shape_after_im2col[0][1], matrix_shape_after_im2col[1][0], 
            matrix_shape_after_im2col[1][1])
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
    2. 由于bram深度为512，所以如果A和B的深度超过了512，就浪费了带宽，所以初始时
    限制A和B的最大深度为512，即单组bram最大支持512x512的矩阵乘法。
    3. 对于bram只能容纳小于等于一组的，直接计算即可
    4. 对于bram能够容纳大于等于1组的，只要完整组。
    5. 保证组的数量为2的幂
    6. 对于bram能够容纳大于1组的，如果资源足够使组的边长和深度均翻倍，则增加单个组
    的大小。即如果能够容纳4个512组，则使组大小变为1024。
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
    bytes_per_bram_line = 8     # bram一行的字节数
    
    matrix_len = 8  # 支持的最大矩阵边长。不能小于8，因为一块bram就能支持8
    bram_group = 0  # 完整的512 bram组数
    while(True):
        # 计算当前matrix_len需要的空间，仅在512以下查找
        space_A_need = matrix_len * matrix_len
        space_B_need = matrix_len * matrix_len
        bram_A_need = matrix_len // bytes_per_bram_line
        bram_B_need = matrix_len // bytes_per_bram_line
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
                xxlog("Device is too small to accelerate neural network", 
                    XXError())
                raise ValueError("Device is too small to accelerate " \
                    "neural network")
            else:
                matrix_len //= 2
            bram_A_need = matrix_len // bytes_per_bram_line
            bram_B_need = matrix_len // bytes_per_bram_line
            space_C_need = matrix_len * matrix_len * 4
            bram_C_need = int(math.ceil(space_C_need / space_per_bram))
            total_bram_need = bram_A_need + bram_B_need + bram_C_need
            break

    # 记录不完整组的长度
    incomplete_bram_group_len = matrix_len

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
            "bram usage: %d, bram avaliable: %d"%(incomplete_bram_group_len, 
            total_bram_need, int(bram_threshold*bram)))
    
    '''
    对完整的bram组进行4合1，并对应增大片上支持的最大矩阵边长，并计算支持的最小矩阵边长
    '''
    # 片上支持的最大矩阵
    max_len_support = max_len_per_group if(bram_group >= 1) \
        else incomplete_bram_group_len
    if(bram_group >= 4):
        xxlog("Try to merge 4 complete group into 1...")
        while(bram_group >= 4):
            # bram_group每4倍，最大矩阵边长2倍
            bram_group //= 4
            max_len_support *= 2
        xxlog("Merge result: Group: %d. Max len support: %d"%(
            bram_group, max_len_support))
    # 片上支持的最小矩阵(保证最小矩阵至少能填满bram组的一行)
    min_len_support = 8
    while(min_len_support**2 < max_len_support):
        min_len_support *= 2
    xxlog("Min len support: %d"%(min_len_support))

    
    '''
    检查计算资源是否足够
    '''
    lut_need_per_mult = 61
    lut_need_per_add = 8
    lut_need_per_sub = 8
    lut_counter_per_dsp = 25    # 每个dsp能够抵消的lut数量(估计值, 不一定准确)
    total_mult = bram_group * max_len_support + incomplete_bram_group_len
    total_add = bram_group * (max_len_support-1) + incomplete_bram_group_len-1
    total_sub = bram_group * max_len_support * 2 + \
        incomplete_bram_group_len * 2
    total_lut_need = (total_mult * lut_need_per_mult + 
        total_add * lut_need_per_add + total_sub * lut_need_per_sub)
    xxlog("Lut need(no consider dsp): %d. Lut avaliable: %d"%(
        total_lut_need, int(lut_threshold*lut)))

    
    calc_unit_per_bram_group = 1
    if(total_lut_need - lut_counter_per_dsp*dsp <= int(lut_threshold*lut)):
        # 如果资源充足，增加每个bram组的计算单元数量
        xxlog("Try to double calculation unit")
        while(True):
            total_lut_need *= 2
            calc_unit_per_bram_group *= 2
            if(total_lut_need - lut_counter_per_dsp*dsp 
                <= int(lut_threshold*lut)):
                xxlog("Calculation unit per bram group: %d, " \
                    "Total lut need: %d. Lut avaliable: %d"%(
                        calc_unit_per_bram_group, total_lut_need,
                        int(lut_threshold*lut)
                    ))
            else:
                # 此时已经用超了
                total_lut_need //= 2
                calc_unit_per_bram_group //= 2
                xxlog("First lut allocate finished: Calculation unit " \
                    "per group: %d. Total lut need: %d. Lut avaliable: %d"%(
                        calc_unit_per_bram_group, total_lut_need, 
                        int(lut_threshold*lut)))
                break
    else:
        # 如果资源不够，进行缩减bram数量
        xxlog("Lut not enough, try to decrease...")
        solved = False
        # 尝试减少完整的bram组
        while(max_len_support > 512 or bram_group > 1):
            if(bram_group == 1):
                # 如果此时是一个大bram组，则拆成4个小的，然后减半
                max_len_support //= 2
                total_lut_need //= 2
                bram_group = 2
            elif(bram_group == 2):
                # 如果此时是2个bram组，则减半。max_len_support不需要变
                bram_group //= 2
                total_lut_need //= 2
            else:
                xxlog("Wrong bram_group value: %d"%(bram_group), XXError())
                raise ValueError("Wrong bram_group value")
            xxlog("Decrease bram group to %d, max_len_support to: %d"%(
                bram_group, max_len_support))
            if(total_lut_need - lut_counter_per_dsp*dsp 
                <= int(lut_threshold*lut)):
                solved = True
                xxlog("Lut enough now, need:%d, avaliable:%d"%(
                    total_lut_need, int(lut_threshold*lut)))
                break
        # 如果bram组不多余1组，尝试缩减最大矩阵边长
        if(not solved):
            while(max_len_support >= 8):
                max_len_support //= 2
                total_lut_need //= 2
                xxlog("Decrease max_len_support to %d"%(max_len_support))
                if(total_lut_need - lut_counter_per_dsp*dsp 
                    <= int(lut_threshold*lut)):
                    solved = True
                    xxlog("Lut enough now, need:%d, avaliable:%d"%(
                        total_lut_need, int(lut_threshold*lut)))
                    break
        # 如果矩阵边长小于8，lut仍不够，报错
        if(not solved):
            xxlog("Device is too small to accelerate neural network", 
                XXError())
            raise ValueError("Device is too small to accelerate " \
                "neural network")
        # 解决后，重新计算bram需求
        bram_A_need = max_len_support // bytes_per_bram_line
        bram_B_need = max_len_support // bytes_per_bram_line
        space_C_need = max_len_support * max_len_support * 4
        bram_C_need = int(math.ceil(space_C_need / space_per_bram))
        total_bram_need = bram_A_need + bram_B_need + bram_C_need
    
    incomplete_bram_group_len = 0 if(bram_group >= 1) else max_len_support
    # 此时初次分配完成
    xxlog("First bram analyse and lut allocate finished, the result is " \
        "shown below:\n" \
        "\tComplete bram group(%d group): %d\n" \
        "\tIncomplete bram group len: %d\n" \
        "\tTotal bram need: %d\n" \
        "\tBram avaliable: %d\n" \
        "\tMax matrix len support: %d\n" \
        "\tMin matrix len support: %d\n" \
        "\tCalculation unit per bram group: %d\n" \
        "\tTotal lut need: %d\n" \
        "\tLut avaliable: %d"%(
            max_len_support, bram_group, incomplete_bram_group_len, 
            total_bram_need, int(bram_threshold*bram), max_len_support, 
            min_len_support, calc_unit_per_bram_group, total_lut_need, 
            int(lut_threshold*lut)
        ))



    '''
    修正C占用空间
    -- 拆分张量表达式
    对矩阵进行切块
    1. 尽量切大块，但不超过片上支持的最大边长
    2. 最小边长不能小于片上支持的最小边长
    3. 矩阵的相乘边要为2的幂
    4. 矩阵的结果边要为合适的长度，使得本矩阵块能够填满bram的一行
        (由于前面三条的限制，本条一定能满足)
    得到切块结果
    '''
    while(True):
        original_shape = []
        xxlog("Copy im2col_shape to original_shape...")
        for shape in im2col_shape:
            original_shape.append(([shape[0][0], shape[0][1]], 
                [shape[1][0], shape[1][1]]))
        # 切分结果。它是一个list。里面的每个元素是一个tuple，表示一层的切分结果。
        # tuple里有两个元素，分别为A和B的切分结果，它们都是list
        # 这两个list中每个包含n个list。每个list包含4个元素，分别为切块在
        # 纵向和横向上的起止点
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
                    [start_h_A, start_h_A+cut_height, 
                        start_w_A, start_w_A+cut_width]
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
                    [start_h_B, start_h_B+cut_height, 
                        start_w_B, start_w_B+cut_width]
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
        # 1. 从切块结果中找到相乘边(A的上边和B的左边)最小的矩阵块
        #   (只需要找A的上边即可)
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
        result_per_cycle_per_bram_group_per_calc_unit = \
            max_len_support // min_matrix_len 
        xxlog("Result get per cycle per bram group per calculation unit: %d"%(
            result_per_cycle_per_bram_group_per_calc_unit))
        # 2. 每个bram_group里C所需要的bram数需要在1的基础上，
        #   再乘上每块里bram分配的计算单元组数。
        # 每组每周期得到的结果数
        result_per_cycle_per_bram_group = (
            result_per_cycle_per_bram_group_per_calc_unit * 
            calc_unit_per_bram_group)
        xxlog("Result per cycle per bram group: %d"%(
            result_per_cycle_per_bram_group))
        # 3. 根据C需要的带宽计算C实际需要的bram数
        result_per_bram = 2 # 结果为32bit的条件下，每列bram每周期能写入的结果数
        # 每个bram_group C需要的bram列数
        bram_col_C_need_per_bram_group = result_per_cycle_per_bram_group \
            // result_per_bram 
        xxlog("Bram col C need per bram group: %d"%(
            bram_col_C_need_per_bram_group))
        # 每个bram_group C需要的空间
        space_C_need_per_bram_group = max_len_support * max_len_support * 4
        xxlog("Space C need per bram group: %d"%(space_C_need_per_bram_group))
        # 每个bram_group C需要每列bram的空间
        space_C_need_per_bram_col = space_C_need_per_bram_group \
            // bram_col_C_need_per_bram_group
        xxlog("Space C need per bram col: %d"%(space_C_need_per_bram_col))
        # C需要每列bram的深度
        depth_C_need_per_bram_col = space_C_need_per_bram_col \
            // bytes_per_bram_line
        xxlog("Depth C need per bram col: %d"%(depth_C_need_per_bram_col))
        # C需要每列bram的个数(由于不知道xilinx bram的计算方式，这一步目前只能查表)
        bram36_C_need_per_col = get_bram_usage(width_per_bram, 
            depth_C_need_per_bram_col)[0]
        bram18_C_need_per_col = get_bram_usage(width_per_bram, 
            depth_C_need_per_bram_col)[1]
        xxlog("Bram36 C need per col: %d, bram18 C need per col: %d"%(
            bram36_C_need_per_col, bram18_C_need_per_col))
        # 每个bram_group C需要的bram个数
        bram_C_need_per_bram_group = (bram_col_C_need_per_bram_group 
            * bram36_C_need_per_col + int(math.ceil(
                bram_col_C_need_per_bram_group * bram18_C_need_per_col / 2)))
        xxlog("Bram C need per bram group: %d"%(bram_C_need_per_bram_group))
        # 每个bram_group ABC总需要的bram个数
        if(bram_group == 0):
            bram_A_need_per_bram_group = max_len_support // bytes_per_bram_line
            bram_B_need_per_bram_group = max_len_support // bytes_per_bram_line
        else:
            bram_A_need_per_bram_group = ((max_len_support 
                // bytes_per_bram_line) * (max_len_support // depth_per_bram))
            bram_B_need_per_bram_group = ((max_len_support 
                // bytes_per_bram_line) * (max_len_support // depth_per_bram))
        xxlog("Bram A need per bram group: %d, Bram B need per bram " \
            "group: %d"%(
                bram_A_need_per_bram_group, bram_B_need_per_bram_group))
        total_bram_need_per_bram_group = (bram_A_need_per_bram_group + 
            bram_B_need_per_bram_group + bram_C_need_per_bram_group)
        xxlog("Total bram need per bram group: %d"%(
            total_bram_need_per_bram_group))
        # bram总数
        if(bram_group == 0):
            total_bram_need = total_bram_need_per_bram_group
        else:
            total_bram_need = total_bram_need_per_bram_group * bram_group
        
        # 如果C修正后bram仍然足够，则可以结束
        if(total_bram_need <= int(bram_threshold*bram)):
            xxlog("Bram enough after fixed C")
            break
        
        # C修正后，bram不足，对bram使用进行修正
        xxlog("Bram not enough after fixed C, try to decrease bram...")
        if(max_len_support > 512 or bram_group > 1):
            if(bram_group == 1):
                bram_group = 2
                max_len_support //= 2
            elif(bram_group == 2):
                bram_group //= 2
        elif(bram_group == 1):
            bram_group = 0
            max_len_support = 512
            max_len_support //= 2
        else:
            max_len_support //= 2
        xxlog("Decrease bram_group to: %d, max_len_support to: %d"%(
            bram_group, max_len_support))
        if(max_len_support < 8):
            xxlog("Device is too small to accelerate neural network", 
                XXError())
            raise ValueError("Device is too small to accelerate " \
                "neural network")
        if(bram_group == 0):
            bram_A_need_per_bram_group = max_len_support // bytes_per_bram_line
            bram_B_need_per_bram_group = max_len_support // bytes_per_bram_line
        else:
            bram_A_need_per_bram_group = ((max_len_support 
                // bytes_per_bram_line) * (max_len_support // depth_per_bram))
            bram_B_need_per_bram_group = ((max_len_support 
                // bytes_per_bram_line) * (max_len_support // depth_per_bram))
        if(bram_group == 0):
            bram_A_need = bram_A_need_per_bram_group
            bram_B_need = bram_B_need_per_bram_group
        else:
            bram_A_need = bram_A_need_per_bram_group * bram_group
            bram_B_need = bram_B_need_per_bram_group * bram_group
        space_C_need = max_len_support * max_len_support * 4
        bram_C_need = int(math.ceil(space_C_need / space_per_bram))
        total_bram_need = bram_A_need + bram_B_need + bram_C_need
        min_len_support = 8
        while(min_len_support**2 < max_len_support):
            min_len_support *= 2
        # 同时对lut使用进行修正
        xxlog("Divide bram usage by 2. Try to fix lut usage...")
        incomplete_bram_group_len = 0 if(bram_group >= 1) else max_len_support
        total_mult = bram_group * max_len_support + incomplete_bram_group_len
        total_add = bram_group * (max_len_support-1) + \
            incomplete_bram_group_len-1
        total_sub = bram_group * max_len_support * 2 + \
            incomplete_bram_group_len * 2
        total_lut_need = (total_mult * lut_need_per_mult + 
            total_add * lut_need_per_add + total_sub * lut_need_per_sub)
        xxlog("Lut need(no consider dsp): %d. Lut avaliable: %d"%(
            total_lut_need, int(lut_threshold*lut)))
        
        calc_unit_per_bram_group = 1
        if(total_lut_need - lut_counter_per_dsp*dsp <= int(lut_threshold*lut)):
            # 如果资源充足，增加每个bram组的计算单元数量
            xxlog("Try to double calculation unit")
            while(True):
                total_lut_need *= 2
                calc_unit_per_bram_group *= 2
                if(total_lut_need - lut_counter_per_dsp*dsp 
                    <= int(lut_threshold*lut)):
                    xxlog("Calculation unit per bram group: %d, " \
                        "Total lut need: %d. Lut avaliable: %d"%(
                            calc_unit_per_bram_group, total_lut_need,
                            int(lut_threshold*lut)
                        ))
                else:
                    # 此时已经用超了
                    total_lut_need //= 2
                    calc_unit_per_bram_group //= 2
                    xxlog("First lut allocate finished: Calculation unit " \
                        "per group: %d. Total lut need: %d. Lut avaliable" \
                        ": %d"%(
                            calc_unit_per_bram_group, total_lut_need, 
                            int(lut_threshold*lut)))
                    break
        incomplete_bram_group_len = 0 if(bram_group >= 1) else max_len_support
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
                bram_group, incomplete_bram_group_len, total_bram_need, 
                int(bram_threshold*bram), max_len_support, min_len_support, 
                calc_unit_per_bram_group, total_lut_need, 
                int(lut_threshold*lut)
            ))
        xxlog("Check again...")

    # 返回结果
    incomplete_bram_group_len = 0 if(bram_group >= 1) else max_len_support
    xxlog("Bram decrease and lut allocate finished, the result is " \
        "shown below:\n" \
        "\tComplete bram group: %d\n" \
        "\tIncomplete bram group len: %d\n" \
        "\tBram column C need per bram group: %d\n" \
        "\tDepth C need per bram col: %d\n" \
        "\tTotal bram need: %d\n" \
        "\tBram avaliable: %d\n" \
        "\tMax matrix len support: %d\n" \
        "\tMin matrix len support: %d\n" \
        "\tCalculation unit per bram group: %d\n" \
        "\tTotal lut need: %d\n" \
        "\tLut avaliable: %d"%(
            bram_group, incomplete_bram_group_len, 
            bram_col_C_need_per_bram_group, depth_C_need_per_bram_col, 
            total_bram_need, int(bram_threshold*bram), 
            max_len_support, min_len_support, calc_unit_per_bram_group, 
            total_lut_need, int(lut_threshold*lut)
            ))

    return {
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
        "calc_unit_per_bram_group": calc_unit_per_bram_group,   
        # 需要的总lut数
        "total_lut_need": total_lut_need,          
         # 可用的lut数
        "lut_avaliable": int(lut_threshold*lut)    
    }


def split_tensor_expression_first_time(
    first_analyse_result,
    im2col_shape,
    calculation_graph
):
    '''
    第一次拆分张量表达式
    3. 第一次拆分张量表达式
    对矩阵进行切块
    1. 尽量切大块，但不超过片上支持的最大边长
    2. 矩阵的相乘边要为2的幂
    3. 矩阵的结果边要为合适的长度，使得本矩阵块能够填满bram的一行
    '''
    xxlog("Split tensor expression first time")

    # 从第一次分析结果中读取max_len_support和min_len_support
    max_len_support = first_analyse_result["max_matrix_len_support"]
    min_len_support = first_analyse_result["min_matrix_len_support"]
    xxlog("max len support: %d, min len support: %d"%(
        max_len_support, min_len_support))
    xxlog("Try to split matrix block with len between %d and %d"%(
        max_len_support, min_len_support))
    
    # 切分矩阵
    original_shape = []
    xxlog("Copy im2col_shape to original_shape...")
    for shape in im2col_shape:
        original_shape.append(([shape[0][0], shape[0][1]], 
            [shape[1][0], shape[1][1]]))
    # 切分结果。它是一个list。里面的每个元素是一个tuple，表示一层的切分结果。
    # tuple里有两个元素，分别为A和B的切分结果，它们都是list
    # 这两个list中每个包含n个list。每个list包含4个元素，分别为切块在
    # 纵向和横向上的起止点
    divided_border = []
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
                [start_h_A, start_h_A+cut_height, 
                    start_w_A, start_w_A+cut_width]
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
                [start_h_B, start_h_B+cut_height, 
                    start_w_B, start_w_B+cut_width]
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
    校验切分结果
    1. 每块边长要为2的幂
    2. A和B的相乘边切分结果要相同
    3. A的不同行的上边切分结果应相同
    4. B的不同列的左边的切分结果应相同
    '''

    xxlog("Checking divide result")

    # 1. 每块边长要为2的幂
    xxlog("Checking: matrix side length should be power of 2")
    for layer in divided_border:
        border_A = layer[0]
        border_B = layer[1]
        for border in border_A:
            matrix_height = border[1] - border[0]
            matrix_width = border[3] - border[2]
            if(not is_power_of_2(matrix_height)):
                xxlog("Check failed: %d is not power of 2"%(
                    matrix_height), XXError())
                raise ValueError("Check failed: %d is not power of 2"%(
                    matrix_width))
            if(not is_power_of_2(matrix_width)):
                xxlog("Check failed: %d is not power of 2"%(
                    matrix_width), XXError())
                raise ValueError("Check failed: %d is not power of 2"%(
                    matrix_width))
        for border in border_B:
            matrix_height = border[1] - border[0]
            matrix_width = border[3] - border[2]
            if(not is_power_of_2(matrix_height)):
                xxlog("Check failed: %d is not power of 2"%(
                    matrix_height), XXError())
                raise ValueError("Check failed: %d is not power of 2"%(
                    matrix_width))
            if(not is_power_of_2(matrix_width)):
                xxlog("Check failed: %d is not power of 2"%(
                    matrix_width), XXError())
                raise ValueError("Check failed: %d is not power of 2"%(
                    matrix_width))
    xxlog("Checking passed")

    # 2. A和B的相乘边结果要相同
    xxlog("Checking the multiply side of A and B and ensure them be the same")
    for layer in divided_border:
        temp_length_A = []
        temp_length_B = []
        border_A = layer[0]
        border_B = layer[1]
        for border in border_A:
            if(border[0] == 0):
                temp_length_A.append(border[3] - border[2])
        for border in border_B:
            if(border[2] == 0):
                temp_length_B.append(border[1] - border[0])
        if(temp_length_A != temp_length_B):
            xxlog("Check failed: %s not equal %s"%(
                temp_length_A, temp_length_B))
            raise ValueError("Check failed: %s not equal %s"%(
                temp_length_A, temp_length_B))
    xxlog("Checking passed")

    # 3. A的不同行的上边切分结果要相同
    xxlog("Checking up side of A and ensure matrix block of different " \
        "row has the same upside len")
    for layer in divided_border:
        temp_length = []
        border_A = layer[0]
        last_start_h = -1
        count = 0
        for border in border_A:
            if(border[0] == 0):     # 记录下第一行
                temp_length.append(border[3] - border[2])
            if(border[0] != last_start_h):  # 到新的一行块数重置
                count = 0
                last_start_h = border[0]
            else:   # 记录后面行当前的块数
                count += 1
            if(border[0] != 0):
                if(temp_length[count] != border[3] - border[2]):
                    xxlog("Check failed: %d not equal %d"%(
                        temp_length[count], border[3] - border[2]))
                    raise ValueError("Check failed: %d not equal %d"%(
                        temp_length[count], border[3] - border[2]))
    xxlog("Checking passed")

    # 4. B的不同列的左边切分结果要相同
    xxlog("Checking left side of B and ensure matrix block of different col " \
        "has the same leftside len")
    for layer in divided_border:
        temp_length = 0
        border_B = layer[1]
        for border in border_B:
            if(border[2] == 0):
                temp_length = border[1] - border[0]
            else:   # 每行后面几个的左边需与第一块一样长
                if(temp_length != border[1] - border[0]):
                    xxlog("Check failed: %d not equal %d"%(
                        temp_length, border[1] - border[0]))
                    raise ValueError("Check failed: %d not equal %d"%(
                        temp_length, border[1] - border[0]))
    xxlog("Checking passed")


    '''
    切分之后，如果切出来的矩阵边长均小于片上支持的最大矩阵边长，则考虑缩减C的空间
    缩减方式：找到一个计算流程，使得
    1. 尽可能多累加。
    2. 传输边长小于等于64的矩阵时尽可能填满A和B。传输边长大于等于128的矩阵时，
        由于每次只需要传输一个矩阵，不需要尽可能填满A和B
    根据该计算流程计算C的峰值占用空间
    '''
    # 查找切块结果中最大矩阵边长(因为是为了检查C的峰值占用空间，所以应该检查A和B的结果边)

    # 1. 因为现在没有限定矩阵一定是方阵，所以仅检查一条边是没用的，
    #   需要同时检查A和B的结果边。但又不知道A和B中子矩阵的对应关系，所以需要先按行列编号，
    #   才能确定A和B中的哪两个矩阵应该对应相乘。

    # 子矩阵的边长。它是一个list。里面的每个元素是一个tuple，表示一层的子矩阵边长
    # tuple里包含两个元素，它们都是list，分别表示A和B的子矩阵边长
    # 这两个list中每个包含多个list，表示A或B一行子矩阵的边长
    # 每个list中包含多个tuple，表示该行每个子矩阵的边长
    # tuple中有两个元素，分别为height和width
    xxlog("Assigning index to divide submatrix...")
    submatrix_size = []
    for n, layer in enumerate(divided_border):
        border_A = layer[0]
        border_B = layer[1]
        submatrix_size_A = []
        submatrix_size_B = []

        # 遍历A
        last_start_h = -1
        submatrix_size_A_current_len = []
        for border in border_A:
            start_h = border[0]
            height = border[1] - border[0]
            width = border[3] - border[2]
            if(start_h != last_start_h):
                # 如果到新的一行
                last_start_h = start_h
                if(start_h != 0):
                    # 如果不是第一行，把上一行的结果加入进去
                    submatrix_size_A.append(submatrix_size_A_current_len)
                # 清空上一行结果
                submatrix_size_A_current_len = []
            submatrix_size_A_current_len.append((height, width))
        # 最后，把最后一行加入进去
        submatrix_size_A.append(submatrix_size_A_current_len)

        # 遍历B
        last_start_h = -1
        submatrix_size_B_current_len = []
        for border in border_B:
            start_h = border[0]
            height = border[1] - border[0]
            width = border[3] - border[2]
            if(start_h != last_start_h):
                # 如果到新的一行
                last_start_h = start_h
                if(start_h != 0):
                    # 如果不是第一行，把上一行的结果加进去
                    submatrix_size_B.append(submatrix_size_B_current_len)
                # 清空上一行结果
                submatrix_size_B_current_len = []
            submatrix_size_B_current_len.append((height, width))
        # 最后，把最后一行加入进去
        submatrix_size_B.append(submatrix_size_B_current_len)

        # 遍历完成后，将当前层结果加入最终结果
        submatrix_size.append((submatrix_size_A, submatrix_size_B))

        xxlog("Current layer size(regard a submatrix block as an element): "\
            "A: [%d,%d], B: [%d,%d]"%(
            len(submatrix_size_A), len(submatrix_size_A[0]), 
            len(submatrix_size_B), len(submatrix_size_B[0])))
    xxlog("Assign finished")

    # 2. 同时检查A和B对应矩阵的结果边的长度。由于如果子矩阵边长达到了最大值，就一定会把
    #   C占满，所以不需要考虑累加的问题，只需要检查是否达到最大值即可。
    #   另一方面，子矩阵边长不可能超过max_len_support，所以最大值为max_len_support
    xxlog("Finding if there is a pair of matrix of A and B to be matmuled " \
        "has the result side with length equal to or larger than " \
        "max_len_support...")
    has_max_matrix = False
    for layer in submatrix_size:
        submatrix_size_A = layer[0]
        submatrix_size_B = layer[1]
        # 将每个子矩阵视为一个元素时，矩阵AB的尺寸
        shape_A = (len(submatrix_size_A), len(submatrix_size_A[0]))
        shape_B = (len(submatrix_size_B), len(submatrix_size_B[0]))
        for i in range(shape_A[0]):
            for j in range(shape_B[1]):
                for k in range(shape_A[1]):
                    # 当前要相乘的子矩阵的尺寸
                    size_A = submatrix_size_A[i][k]
                    size_B = submatrix_size_B[k][j]
                    if(size_A[0] >= max_len_support and 
                        size_B[1] >= max_len_support):
                        has_max_matrix = True
                        break
    xxlog("Found: %s"%(has_max_matrix))

    # 如果找到了结果边均达到max_len_support的矩阵，说明C一定是占满的
    if(has_max_matrix):
        xxlog("Since matrix with side length equal to or larger than " \
            "max_len_support is found, C is full used, and the function " \
            "should return")
        return {
            # C是否被占满
            "is_c_fulled_used": True,
            # C的最大使用量
            "c_max_usage": None
        }


    # 如果没找到结果边均达到max_len_support的矩阵，说明C可能无法占满
    # 寻找C的最大使用量
    xxlog("Founding the max usage of C")
    C_max_usage = 0
    for layer_index, layer in enumerate(submatrix_size):
        # 将当前层拆分为张量表达式
        # C_0_0 = A_0_0 * B_0_0 + A_0_1 * B_1_0 + ...
        xxlog("Dividing current layer into tensor expression")
        submatrix_size_A = layer[0]
        submatrix_size_B = layer[1]
        tensor_expr = []
        shape_A = (len(submatrix_size_A), len(submatrix_size_A[0]))
        shape_B = (len(submatrix_size_B), len(submatrix_size_B[0]))
        M = shape_A[0]
        N = shape_B[1]
        K = shape_A[1]
        for m in range(M):
            for n in range(N):
                string = "C_%d_%d="%(m, n)
                for k in range(K):
                    if(k == 0):
                        string += "A_%d_%d*B_%d_%d"%(m, k, k, n)
                    else:
                        string += "+A_%d_%d*B_%d_%d"%(m, k, k, n)
                tensor_expr.append(string)
        xxlog("Dividing tensor expression finished: %s"%(tensor_expr))

        # 目前当前层已经拆分为张量表达式，规划其计算流程
        # 原则上，数据量大于等于16384(128x128)的矩阵单传更划算，
        #   小于该值的合并传更划算
        # 那么考虑以下四种情况
        # 1. 连续的小矩阵，攒到16384或填满ABC中的一个停止
        # 2. 连续的大矩阵。单独传输
        # 3. 连续的小矩阵后是大矩阵。若小矩阵不够16384且空间足够，尝试将大矩阵加入进去
        # 4. 大矩阵后是连续的小矩阵。如果小矩阵数量少且大矩阵未达最大限制，尝试加入进去
        # 在规划过程中，一定不跨层规划。对于跨C块规划问题：如果当前容量能够将下一个C块
        # 完整地容纳进来，则容纳，否则不跨C块。

        # 当前一次传输中，计划传输的A矩阵和B矩阵。完成一次传输就要清空
        # 以及计划计算的C矩阵
        A_in_plan = []
        B_in_plan = []
        C_in_plan = []
        calc_in_plan = []
        # 当前传输规划中，ABC已经累积的空间
        A_accumulate_space = 0
        B_accumulate_space = 0
        C_accumulate_space = 0
        # ABC的容量
        bram_group = first_analyse_result["bram_group"]
        depth_per_bram = 512
        bram_col_C_need_per_bram_group = first_analyse_result[
            "bram_col_c_need_per_bram_group"]
        depth_c_need_per_bram_col = first_analyse_result[
            "depth_c_need_per_bram_col"]
        A_capacity = max_len_support*depth_per_bram if(max_len_support < 512) \
            else (max_len_support*max_len_support*bram_group)
        B_capacity = max_len_support*depth_per_bram if(max_len_support < 512) \
            else (max_len_support*max_len_support*bram_group)
        C_capacity = bram_col_C_need_per_bram_group * \
            depth_c_need_per_bram_col * 8 if(bram_group == 0) else \
            bram_col_C_need_per_bram_group * depth_c_need_per_bram_col * \
            bram_group * 8
        xxlog("Got the capaticy of A, B, C: %d, %d, %d"%(
            A_capacity, B_capacity, C_capacity))
        xxlog("im2col shape: A: %s, B: %s"%(original_shape[layer_index][0], 
            original_shape[layer_index][1]))
        # 按照传输规划的张量表达式
        tensor_expr_with_transport_plan = []

        def split_adder(adder):
            # 拆分adder分析数据
            multer_A = adder.split("*")[0]
            multer_B = adder.split("*")[1]
            index_A = [int(i) for i in multer_A.split("_")[1:3]]
            index_B = [int(i) for i in multer_B.split("_")[1:3]]
            size_A = submatrix_size_A[index_A[0]][index_A[1]]
            size_B = submatrix_size_B[index_B[0]][index_B[1]]
            size_C = (size_A[0], size_B[1])
            space_A = size_A[0] * size_A[1]
            space_B = size_B[0] * size_B[1]
            space_C = size_C[0] * size_C[1] * 4
            return multer_A, multer_B, index_A, index_B, size_A, size_B, \
                size_C, space_A, space_B, space_C
            
        def check_space(multer_A, multer_B, block_C, 
            space_A, space_B, space_C):
            # 检查加入当前矩阵后片上空间是否足够
            A_capacity_need = A_accumulate_space
            if(multer_A not in A_in_plan):
                A_capacity_need += space_A
            B_capacity_need = B_accumulate_space
            if(multer_B not in B_in_plan):
                B_capacity_need += space_B
            C_capacity_need = C_accumulate_space
            if(block_C not in C_in_plan):
                C_capacity_need += space_C
            if(A_capacity_need <= A_capacity and 
               B_capacity_need <= B_capacity and
               C_capacity_need <= C_capacity):
                return True
            return False
        
        def clear_plan():
            # 清空计划表，并统计C的峰值占用(不清空C)
            nonlocal A_in_plan, B_in_plan, C_in_plan, calc_in_plan
            nonlocal A_accumulate_space, B_accumulate_space, C_accumulate_space
            nonlocal tensor_expr_with_transport_plan
            nonlocal C_max_usage
            for i in A_in_plan:
                tensor_expr_with_transport_plan.append("load " + i)
            for i in B_in_plan:
                tensor_expr_with_transport_plan.append("load " + i)
            for i in calc_in_plan:
                tensor_expr_with_transport_plan.append(i)
            A_in_plan = []
            B_in_plan = []
            calc_in_plan = []
            C_max_usage = max(C_max_usage, C_accumulate_space)
            A_accumulate_space = 0
            B_accumulate_space = 0
            xxlog("Clear plan. C max usage: %d. Now tensor_expr: %s"%(
                C_max_usage, tensor_expr_with_transport_plan))
        
        def write_back_C():
            nonlocal C_in_plan, C_accumulate_space
            for i in C_in_plan:
                tensor_expr_with_transport_plan.append("store " + i)
            C_in_plan = []
            C_accumulate_space = 0
            xxlog("Write back C. Now tensor_expr: %s"%(
                tensor_expr_with_transport_plan))

        def check_can_hold_next_C_block_all(C_block_index):
            # 检查在当前情况的基础上，是否能把计算下一个C块(先检查是否还有下一个C块)
            # 需要的内容全部存进来，且要求计算下一个C块中不含大矩阵，且B的占用不超65536
            find_index = C_block_index + 1
            if(find_index >= len(tensor_expr)):
                # 没有下一个C块了，要求清空plan
                return False
            temp_tensor_expr = tensor_expr[find_index]
            block_C = temp_tensor_expr.split("=")[0]
            adders_str = temp_tensor_expr.split("=")[1]
            adders = adders_str.split("+")
            A_in_plan_copy = [i for i in A_in_plan]
            B_in_plan_copy = [i for i in B_in_plan]
            C_in_plan_copy = [i for i in C_in_plan]
            A_accumulate_space_copy = A_accumulate_space
            B_accumulate_space_copy = B_accumulate_space
            C_accumulate_space_copy = C_accumulate_space
            for adder in adders:
                multer_A, multer_B, index_A, index_B, size_A, size_B, size_C, \
                space_A, space_B, space_C = split_adder(adder)
                if(is_large_matrix(space_B)):
                    # 如果含大矩阵，要求清空plan
                    return False
                if(multer_A not in A_in_plan_copy):
                    A_in_plan_copy.append(multer_A)
                    A_accumulate_space_copy += space_A
                if(multer_B not in B_in_plan_copy):
                    B_in_plan_copy.append(multer_B)
                    B_accumulate_space_copy += space_B
                if(block_C not in C_in_plan_copy):
                    C_in_plan_copy.append(block_C)
                    C_accumulate_space_copy += space_C
                if(A_accumulate_space_copy > A_capacity):
                    return False
                if(B_accumulate_space_copy > B_capacity or
                   B_accumulate_space_copy > 65536):
                    return False
                if(C_accumulate_space_copy > C_capacity):
                    return False
            return True


        for C_block_index, temp_tensor_expr in enumerate(tensor_expr):
            xxlog("Traverse each tensor expression in current layer: %s"%(
                temp_tensor_expr))
            block_C = temp_tensor_expr.split("=")[0]
            adders_str = temp_tensor_expr.split("=")[1]
            adders = adders_str.split("+")
            C_first_access = True

            def add_to_plan(multer_A, multer_B, block_C, size_A, size_B,
                size_C, space_A, space_B, space_C):
                # 将当前块加入plan中
                nonlocal A_in_plan, B_in_plan, C_in_plan, calc_in_plan
                nonlocal A_accumulate_space, B_accumulate_space
                nonlocal C_accumulate_space, C_first_access
                if(multer_A not in A_in_plan):
                    A_in_plan.append(multer_A)
                    A_accumulate_space += space_A
                if(multer_B not in B_in_plan):
                    B_in_plan.append(multer_B)
                    B_accumulate_space += space_B
                if(block_C not in C_in_plan):
                    C_in_plan.append(block_C)
                    C_accumulate_space += space_C
                if(C_first_access):
                    C_first_access = False
                    expr = block_C + "=" + multer_A + "*" + multer_B
                else:
                    expr = block_C + "+=" + multer_A + "*" + multer_B
                if(expr not in calc_in_plan):
                    calc_in_plan.append(expr)
                xxlog("Add %s, %s, %s to plan. sizeA: %s, sizeB: %s, sizeC: " \
                    "%s. spaceA:%d, spaceB:%d, spaceC:%d. usedA:%d, usedB:%d" \
                    " usedC:%d. capacityA:%d, capacityB:%d, capacityC:%d, " \
                    "now planA: %s, planB: %s, planC: %s, plan_calc: %s"%(
                        multer_A, multer_B, block_C, size_A, size_B, size_C, 
                        space_A, space_B, space_C, A_accumulate_space,
                        B_accumulate_space, C_accumulate_space, A_capacity,
                        B_capacity, C_capacity, A_in_plan, B_in_plan, 
                        C_in_plan, calc_in_plan))
            
            def check_can_hold_all(adder_index):
                # 检查下一个大矩阵或当前C块结束前的所有小矩阵是否能够全部装下，
                # 且不超65536
                A_in_plan_copy = [i for i in A_in_plan]
                B_in_plan_copy = [i for i in B_in_plan]
                C_in_plan_copy = [i for i in C_in_plan]
                A_accumulate_space_copy = A_accumulate_space
                B_accumulate_space_copy = B_accumulate_space
                C_accumulate_space_copy = C_accumulate_space
                find_index = adder_index + 1
                while(find_index < len(adders)):
                    adder = adders[find_index]
                    multer_A, multer_B, index_A, index_B, size_A, size_B, \
                    size_C, space_A, space_B, space_C = split_adder(adder)
                    if(is_large_matrix(space_B)):
                        break
                    if(multer_A not in A_in_plan_copy):
                        A_in_plan_copy.append(multer_A)
                        A_accumulate_space_copy += space_A
                    if(multer_B not in B_in_plan_copy):
                        B_in_plan_copy.append(multer_B)
                        B_accumulate_space_copy += space_B
                    if(block_C not in C_in_plan_copy):
                        C_in_plan_copy.append(block_C)
                        C_accumulate_space_copy += space_C
                    if(A_accumulate_space_copy > A_capacity or 
                       B_accumulate_space_copy > B_capacity or 
                       C_accumulate_space_copy > C_capacity or 
                       B_accumulate_space_copy > 65536):
                        return False
                    find_index += 1
                return True

            for adder_index, adder in enumerate(adders):
                xxlog("Traverse each pair of adders in current tensor " \
                    "expression: %d: %s"%(adder_index, adder))
                multer_A, multer_B, index_A, index_B, size_A, size_B, size_C, \
                space_A, space_B, space_C = split_adder(adder)
                if(is_large_matrix(space_B)):
                    # 如果B是大矩阵
                    if(len(A_in_plan) != 0 or len(B_in_plan) != 0):
                        # 如果计划表不空
                        can_hold = check_space(multer_A, multer_B, 
                            block_C, space_A, space_B, space_C)
                        if(can_hold):
                            # 如果空间足够
                            add_to_plan(multer_A, multer_B, block_C, size_A, 
                                size_B, size_C, space_A, space_B, space_C)
                            clear_plan()
                            continue
                        else:
                            # 如果空间不够
                            clear_plan()
                            # 下一步交给计划表空的情况处理
                    if(len(A_in_plan) == 0 and len(B_in_plan) == 0):
                        # 如果计划表空(为了接住上面一行的结果，这里不用else)
                        add_to_plan(multer_A, multer_B, block_C, size_A,
                            size_B, size_C, space_A, space_B, space_C)
                        if(adder_index+1 < len(adders)):
                            # 如果还有下一对AB矩阵
                            next_adder = adders[adder_index+1]
                            multer_A_next, multer_B_next, index_A_next, \
                            index_B_next, size_A_next, size_B_next, \
                            size_C_next, space_A_next, space_B_next, \
                            space_C_next = split_adder(next_adder)
                            if(is_large_matrix(space_B_next)):
                                # 如果下一个B是大矩阵
                                clear_plan()
                                continue
                            else:
                                # 如果下一对不是大矩阵
                                continue
                        else:
                            # 如果没有下一对AB矩阵了
                            clear_plan()
                            continue
                else:
                    # 如果AB是小矩阵
                    if(len(A_in_plan) == 0 and len(B_in_plan) == 0):
                        # 如果计划表空
                        add_to_plan(multer_A, multer_B, block_C, size_A, 
                            size_B, size_C, space_A, space_B, space_C)
                        continue
                    else:
                        # 如果计划表不空
                        can_hold = check_space(multer_A, multer_B, block_C, 
                            space_A, space_B, space_C)
                        if(can_hold):
                            # 如果空间足够
                            add_to_plan(multer_A, multer_B, block_C, size_A,
                                size_B, size_C, space_A, space_B, space_C)
                            if(B_accumulate_space >= 16384):
                                # 如果B使用容量达16384
                                can_hold_all = check_can_hold_all(adder_index)
                                if(can_hold_all):
                                    # 如果能容纳后面的一系列小矩阵
                                    continue
                                else:
                                    # 如果不能容纳后面的一系列小矩阵
                                    clear_plan()
                                    continue
                            else:
                                continue
                        else:
                            # 如果空间不够
                            clear_plan()
                            add_to_plan(multer_A, multer_B, block_C, size_A,
                                size_B, size_C, space_A, space_B, space_C)
                            continue
            

            # 遍历计算当前C块需要的每一对AB完成
            can_hold_next_C_block_all = check_can_hold_next_C_block_all(
                C_block_index)
            if(can_hold_next_C_block_all):
                continue
            else:
                clear_plan()
                write_back_C()
                continue
    
    
    C_fulled_used = True if(C_max_usage >= C_capacity) else False
    xxlog("First split tensor expression result: \n" \
        "\tis_c_fulled_used: %s\n" \
        "\tc_max_usage: %d"%(C_fulled_used,
        C_max_usage))
    return {
        # C是否被占满
        "is_c_fulled_used": C_fulled_used,
        # C的最大使用量
        "c_max_usage": C_max_usage
    }


def analyse_resources_second_time(
    project_part,
    lut,
    ff,
    bram,
    dsp,
    bram_threshold,
    lut_threshold,
    im2col_shape,
    calculation_graph,
    first_analyse_result,
    first_tensor_expression
):
    xxlog("Analysing resource second time...")

    is_c_fulled_used = first_tensor_expression["is_c_fulled_used"]
    if(True or is_c_fulled_used):
        # 如果C已经占满，则返回原来结果以及不需要更激进的分配
        first_analyse_result_copy = first_analyse_result.copy()
        first_analyse_result_copy["more_radical_allocation"] = False
        xxlog("Found C fully used. Keep old allocation. No more radical " \
            "allocation. The result is shown below:\n" \
            "\tComplete bram group: %d\n" \
            "\tBram column C need per bram group: %d\n" \
            "\tDepth C need per bram col: %d\n" \
            "\tTotal bram need: %d\n" \
            "\tBram avaliable: %d\n" \
            "\tMax matrix len support: %d\n" \
            "\tMin matrix len support: %d\n" \
            "\tCalculation unit per bram group: %d\n" \
            "\tTotal lut need: %d\n" \
            "\tLut avaliable: %d\n" \
            "\tMore radical allocation: %s"%(
            first_analyse_result_copy["bram_group"], 
            first_analyse_result_copy["bram_col_c_need_per_bram_group"], 
            first_analyse_result_copy["depth_c_need_per_bram_col"], 
            first_analyse_result_copy["total_bram_need"], 
            first_analyse_result_copy["bram_avaliable"], 
            first_analyse_result_copy["max_matrix_len_support"], 
            first_analyse_result_copy["min_matrix_len_support"], 
            first_analyse_result_copy["calc_unit_per_bram_group"], 
            first_analyse_result_copy["total_lut_need"], 
            first_analyse_result_copy["lut_avaliable"],
            first_analyse_result_copy["more_radical_allocation"]))
        return first_analyse_result_copy