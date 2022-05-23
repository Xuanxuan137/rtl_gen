#!/bin/bash
python3 main.py \
--model_dir model/mnist_conv3_quanted_output \
--float_model_dir model/mnist_conv3_output \
--part=xc7z020clg400-4 \
--LUT=53200 \
--FF=106400 \
--BRAM=140 \
--DSP=220 \
--data_on_chip=True \
--try_increase_c_bandwidth=True \
--optimize=2

python3 main.py \
--model_dir model/resnet18_quanted_output \
--float_model_dir model/resnet18_output \
--part=xc7z020clg400-4 \
--LUT=53200 \
--FF=106400 \
--BRAM=140 \
--DSP=220 \
--data_on_chip=True \
--try_increase_c_bandwidth=True \
--optimize=2

python3 main.py \
--model_dir model/resnet50_quanted_output \
--float_model_dir model/resnet50_output \
--part=xc7z020clg400-4 \
--LUT=53200 \
--FF=106400 \
--BRAM=140 \
--DSP=220 \
--data_on_chip=True \
--try_increase_c_bandwidth=True \
--optimize=2

python3 main.py \
--model_dir model/vgg11_quanted_output \
--float_model_dir model/vgg11_output \
--part=xc7z020clg400-4 \
--LUT=53200 \
--FF=106400 \
--BRAM=140 \
--DSP=220 \
--data_on_chip=True \
--try_increase_c_bandwidth=True \
--optimize=2