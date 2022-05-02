

import conv
import fc
import post_process
import numpy as np


if __name__ == "__main__":

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
    # print(code)

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
    # print(code)

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
    # print(code)