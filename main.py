

import conv
import fc
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
        DO_RELU=False,
        DEBUG=True,
    )
    # print(code)