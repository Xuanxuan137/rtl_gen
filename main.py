

import math
import argparse

import numpy as np

import conv
import fc
import post_process


def analyse_resources(
    part: str,
    lut: int,
    ff: int,
    bram: int,
    dsp: int,
):
    '''
    此时假设只进行方阵计算
    对于矩阵A，设希望提供的带宽为$8n$，则：
    当$n<=64$时，A和B各需要n块BRAM
    从1开始遍历n
    A，B各占用n块bram。A的数据量为$(8n)^2=64n^2Bytes$，则C的数据量为A的4倍，即$256n^2Bytes$。
    需要的bram数量为$256n^2B/36kb=256n^2B/4.5kB=256n^2/4608=ceil(n^2/18)$块
    当$n>64\ and\ n<=128$时，由于深度超过了单块bram的极限512，所以矩阵的每8列都需要2块bram。
    即便把所有列合在一起申请也是一样，因为为了保证带宽，就没法利用每块里空闲的空间。
    所以A和B各需要2n块BRAM。C不变

    综上：
    A和B各需要的bram数为：
    $$ceil(n/64)\times n$$
    C需要的bram数为：
    $$ceil(n^2/18)$$
    按照
    $$A+B+C<=[0.9*Total]$$
    找到第一次决定的n
    然后根据n判断计算资源是否足够
    如果不够，则从n至1遍历，知道找到合适的n
    '''
    # 对于已经知道的芯片
    if(part == "xc7z020clg400-1" or 
       part == "xc7z020clg400-2" or
       part == "xc7z020clg400-3"):
        result_dict = {}
        result_dict["A_bram_block_number"] = 32         # 在verilog创建的bram实例数
        result_dict["A_width_per_block"] = 64           # 创建bram时的宽度
        result_dict["A_depth_per_block"] = 512          # 创建bram时的深度
        result_dict["B_bram_block_number"] = 32
        result_dict["B_width_per_block"] = 64
        result_dict["B_depth_per_block"] = 512
        result_dict["C_bram_block_number"] = 8
        result_dict["C_width_per_block"] = 64
        result_dict["C_depth_per_block"] = 4096
        result_dict["max_matrix"] = 256                 # 支持的最大矩阵边长
        result_dict["min_matrix"] = 16                  # 支持的最小矩阵边长
        return result_dict
    
    # 对于未知的芯片




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RTL")

    parser.add_argument("-p", "--part", required=True ,help="Project part, e.g. xc7z020clg400-1")
    parser.add_argument("--LUT", help="The number of LUT")
    parser.add_argument("--FF", help="The number of Flip Flop")
    parser.add_argument("--BRAM", help="The number of BRAM")
    parser.add_argument("--DSP", help="The number of DSP")
    parser.add_argument("--data_on_chip", default=True, help="Put part of data on chip")

    args = parser.parse_args()

    project_part = args.part
    lut = args.LUT
    ff = args.FF
    bram = args.BRAM
    dsp = args.DSP
    data_on_chip = args.data_on_chip

    known_parts = [
        "xc7z020clg400-1", "xc7z020clg400-2", "xc7z020clg400-3",
    ]

    if(project_part not in known_parts):
        if(lut is None or
           ff is None or
           bram is None or
           dsp is None):
            raise ValueError("Since part is Unknown, you must specify the number of LUT, FF, BRAM, DSP")
        
    if(data_on_chip == False):
        raise ValueError("data_on_chip=False is not supported yet")

    # 根据型号和片上资源分析buffer和计算单元使用方式
    analyse_result = analyse_resources(
        project_part,
        lut,
        ff,
        bram,
        dsp,
    )
    print(analyse_result);exit()

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