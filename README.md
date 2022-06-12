# 参数说明
--model_dir model/mnist_quanted_output      模型文件夹
--part=xc7z020clg400-4                      目标芯片
--LUT=53200                                 目标芯片lut总量
--FF=106400                                 目标芯片flip flop总量
--BRAM=140                                  目标芯片bram总量
--DSP=220                                   目标芯片dsp总量
--BRAM_threshold=0.9                        bram使用限制
--LUT_threshold=0.7                         lut使用限制
--data_on_chip=True                         是否在芯片上存储一些数据
--try_increase_c_bandwidth=True             是否尝试增加C的带宽以增加性能，这可能导致bram用超
--optimize=2                                是否进行优化(0:不优化, 1:压缩C的空间分给AB使用, 2:更激进地压缩C。优化不一定提升性能)




