

import conv



if __name__ == "__main__":
    code = conv.gen_conv(
        MODULE_NAME="conv",
        MUX_WIDTH=2,
        DATA_WIDTH=8,
        DATA_NUMBER=256,
        OUTPUT_PORTS=[16, 32, 64, 128, 256],
        ZERO_X=[0, 30, 76],
        ZERO_W=[54, 118, 121],
        DEBUG=True,
    )
    print(code)