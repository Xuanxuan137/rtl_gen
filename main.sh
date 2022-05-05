#!/bin/bash
python3 main.py \
--model_dir mnist_quanted_output \
--part=xc7z020clg400-4 \
--LUT=53200 \
--FF=106400 \
--BRAM=545 \
--DSP=220 \
--data_on_chip=True