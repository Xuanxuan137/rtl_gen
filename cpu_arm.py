

import math

import op
import os
import numpy as np



def decimal_to_bin(value, width):
    '''
    transfer decimal value to binary with 'width' width
    '''
    if(value < 0):
        raise ValueError("value should be non-negative")
    code = ""
    for i in range(width):
        if(value >= 2**(width-1-i)):
            code += "1"
            value -= 2**(width-1-i)
        else:
            code += "0"
    return code



def bin_to_hex(bin):
    '''
    convert binary to hex
    '''
    table = {
        "0000": "0",
        "0001": "1",
        "0010": "2",
        "0011": "3",
        "0100": "4",
        "0101": "5",
        "0110": "6",
        "0111": "7",
        "1000": "8",
        "1001": "9",
        "1010": "A",
        "1011": "B",
        "1100": "C",
        "1101": "D",
        "1110": "E",
        "1111": "F",
    }

    while(len(bin) < math.ceil(len(bin)/4)*4):
        bin = "0" + bin
    
    code = ""
    section = len(bin) // 4
    
    for i in range(section):
        part = bin[i*4 : i*4+4]
        code += table[part]
    
    return code



def get_conv2d_index(n, calculation_graph):
    '''
    find the n^th node in calulation_graph is the x^th conv2d, and return x
    count from 0
    '''
    if(type(calculation_graph[n]) != op.Conv2d and 
        type(calculation_graph[n]) != op.QConv2d):
        raise TypeError("Calculation_graph[n] is not Conv2d")
    i = 0
    count = -1
    while(i <= n):
        if(type(calculation_graph[i]) == op.Conv2d or 
            type(calculation_graph[i]) == op.QConv2d):
            count += 1
        i += 1
    return count



def get_node_index(n, calculation_graph):
    '''
    find the n^th conv2d in calculation_graph is the x^th node, and return x
    count from 0
    '''
    count = -1
    for node_index, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or 
            type(node) == op.QConv2d):
            count += 1
            if(count == n):
                return node_index
    return -1



def gen_fixed_point_lib():
    '''
    Generate fixed_point_lib
    '''
    h = \
"""
#ifndef __FIXED_POINT_H__
#define __FIXED_POINT_H__

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>


typedef struct Fixed_point {
    char sign;                      // ????????? 0?????????1??????
    unsigned int ivalue;            // 32???????????????
    unsigned int fvalue;            // 16??????????????????????????????16bit
    void (*assign)(struct Fixed_point * this, float f);
    void (*add)(struct Fixed_point * this, struct Fixed_point * f);
    void (*mult)(struct Fixed_point * this, struct Fixed_point * f);
    int (*to_int)(struct Fixed_point * this);
    void (*print)(struct Fixed_point * this);
    void (*free)(struct Fixed_point * this);
} Fixed_point;

Fixed_point* Fixed_point_init(float f);
void Fixed_point_assign(struct Fixed_point * this, float f);
void Fixed_point_add(struct Fixed_point * this, struct Fixed_point * f);
void Fixed_point_mult(struct Fixed_point * this, struct Fixed_point * f);
int Fixed_point_to_int(struct Fixed_point * this);
void Fixed_point_print(struct Fixed_point * this);
void Fixed_point_free(struct Fixed_point * this);


#endif
"""
    c = \
"""
#include "fixed_point.h"

Fixed_point* Fixed_point_init(float f)
{
    Fixed_point * this = (Fixed_point*)malloc(sizeof(Fixed_point));

    this->assign = &Fixed_point_assign;
    this->add = &Fixed_point_add;
    this->mult = &Fixed_point_mult;
    this->to_int = &Fixed_point_to_int;
    this->print = &Fixed_point_print;
    this->free = &Fixed_point_free;

    this->assign(this, f);

    return this;
}


void Fixed_point_assign(Fixed_point * this, float f)
{
    if(f < -2147483647.0f) {
        printf("assigned value should not be less than -2147483647\\n");
        return;
    }
    else if(f > 2147483647.0f) {
        printf("assigned value should not be larger than 2147483647\\n");
        return;
    }

    this->sign = (f < 0) ? 1 : 0;

    f = fabsf(f);
    unsigned int int_part = (int)f;     // ?????????????????????????????????
    float frac_part = f - int_part;     // ?????????????????????????????????

    this->ivalue = int_part;

    float frac_temp = 1.0;
    float frac_value = 0;               // ??????????????????????????????????????????
    for(int i = 15; i>=0; i--) {
        frac_temp /= 2;
        if(frac_value + frac_temp > frac_part) {
            this->fvalue = (this->fvalue) << 1;
            this->fvalue = (this->fvalue) & 0xfffffffe;
        }
        else {
            this->fvalue = (this->fvalue) << 1;
            this->fvalue = (this->fvalue) | 0x00000001;
            frac_value += frac_temp;
        }
    }
}


void Fixed_point_add(Fixed_point * this, Fixed_point * f)
{
    unsigned int this_fvalue = this->fvalue;
    unsigned int this_ivalue = this->ivalue;
    if(this->sign) {        
        this_fvalue = ~(this->fvalue | 0xffff0000) + 1;    // ??????0xffff0000??????fvalue??????16???(?????????16???)??????1???????????????????????????0????????????1
        int c = (this_fvalue >> 16) & 0x1;     // ??????????????????????????????
        this_fvalue = this_fvalue & 0x0000ffff; // ????????????16???
        this_ivalue = ~(this->ivalue) + c;     // ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    }
    unsigned int f_fvalue = f->fvalue;
    unsigned int f_ivalue = f->ivalue;
    if(f->sign) {
        f_fvalue = ~(f->fvalue | 0xffff0000) + 1;
        int c = (f_fvalue >> 16) & 0x1;
        f_fvalue = f_fvalue & 0x0000ffff;
        f_ivalue = ~(f->ivalue) + c;
    }

    this_fvalue += f_fvalue;
    int c = (this_fvalue >> 16) & 0x1;
    this_fvalue = this_fvalue & 0x0000ffff;
    this_ivalue += f_ivalue + c;

    char this_sign = 0;
    if(this_ivalue & 0x80000000) {  // ??????????????????1
        this_fvalue = this_fvalue - 1;
        int c = (this_fvalue >> 16) & 0x1;      // ?????????????????????
        this_fvalue = (~this_fvalue) & 0x0000ffff;
        this_ivalue = ~(this_ivalue - c);
        this_sign = 1;
    }
    this->fvalue = this_fvalue;
    this->ivalue = this_ivalue;
    this->sign = this_sign;
}



void Fixed_point_mult(Fixed_point * this, Fixed_point * f)
{
    unsigned long long llthis = this->ivalue;
    llthis = llthis << 16;
    llthis = llthis | (this->fvalue & 0x0000ffff);

    unsigned long long f0 = f->fvalue & 0x0000ffff;
    unsigned long long f1 = f->ivalue & 0x0000ffff;
    unsigned long long f2 = ((f->ivalue) >> 16) & 0x0000ffff;

    unsigned long long t0 = llthis * f0;
    unsigned long long t1 = llthis * f1;
    unsigned long long t2 = llthis * f2;

    unsigned char temp0[12] = {0};     
    unsigned char temp1[12] = {0};
    unsigned char temp2[12] = {0};
    for(int i = 0; i<8; i++) {
        temp0[i] = t0 & 0xff;
        t0 >>= 8;
    }
    for(int i = 2; i<10; i++) {
        temp1[i] = t1 & 0xff;
        t1 >>= 8;
    }
    for(int i = 4; i<12; i++) {
        temp2[i] = t2 & 0xff;
        t2 >>= 8;
    }

    unsigned char res[12] = {0};
    int s = 0, c = 0;
    for(int i = 0; i<12; i++) {
        s = temp0[i] + temp1[i] + temp2[i] + c;
        c = s >> 8;
        s = s & 0xff;
        res[i] = s;
    }

    this->ivalue = 0;
    this->fvalue = 0;
    for(int i = 7; i>=4; i--) {
        this->ivalue <<= 8;
        this->ivalue = (this->ivalue & 0xffffff00) | ((unsigned int)res[i] & 0x000000ff);
    }
    for(int i = 3; i>=2; i--) {
        this->fvalue <<= 8;
        this->fvalue = (this->fvalue & 0xffffff00) | ((unsigned int)res[i] & 0x000000ff);
    }
    this->sign = (this->sign) ^ (f->sign);
}

int Fixed_point_to_int(struct Fixed_point * this)
{
    return (this->sign == 1) ? (-(this->ivalue)) : (this->ivalue);
}


void Fixed_point_print(Fixed_point * this)
{
    printf("binary value: ");
    if(this->sign) {
        printf("-");
    }
    for(int i = 31; i>=0; i--) {
        printf("%x", ((this->ivalue) >> i) & 0x1);
    }
    printf(".");
    for(int i = 15; i>=0; i--) {
        printf("%x", ((this->fvalue) >> i) & 0x1);
    }
    printf("\\n");

    printf("decimal value: ");
    if(this->sign) {
        printf("-");
    }
    printf("%u.", this->ivalue);
    float temp = 0;
    for(int i = 15; i>=0; i--) {
        if(((this->fvalue >> i) & 0x1) == 1) {
            temp += pow(2, i-16);
        }
    }
    while(temp > 0) {
        temp *= 10;
        printf("%d", (int)temp);
        temp -= (int)temp;
    }
    printf("\\n");
}


void Fixed_point_free(Fixed_point * this)
{
    free(this);
}
"""
    return h, c



def gen_intnn():
    '''
    Generate int nn calc lib
    '''
    code = \
"""
typedef unsigned char uint8;
void qinput(uint8 * y, uint8 * x, int len)
{
    memcpy(y, x, sizeof(uint8)*len);
}

void qpadding(uint8 * y, uint8 * x, int xn, int xc, int xh, int xw, int ph, int pw, int zero)
{
    int padded_w = xw + 2*pw;
    uint8 * y_ptr = y;
    uint8 * x_ptr = x;
    for(int n = 0; n<xn; n++) {
        for(int c = 0; c<xc; c++) {
            for(int h = 0; h<ph; h++) {
                memset(y_ptr, zero, sizeof(uint8)*padded_w);
                y_ptr += padded_w;
            }
            for(int h = 0; h<xh; h++) {
                memset(y_ptr, zero, sizeof(uint8)*pw);
                y_ptr += pw;
                memcpy(y_ptr, x_ptr, sizeof(uint8)*xw);
                y_ptr += xw;
                x_ptr += xw;
                memset(y_ptr, zero, sizeof(uint8)*pw);
                y_ptr += pw;
            }
            for(int h = 0; h<ph; h++) {
                memset(y_ptr, zero, sizeof(uint8)*padded_w);
                y_ptr += padded_w;
            }
        }
    }
}

void qim2col(uint8 * data_col, uint8 * data_im, int height, int width, int channels_col, 
            int height_col, int width_col, int kernel_h, int kernel_w, int stride_h, int stride_w, 
            int pad_h, int pad_w, int dilation_h, int dilation_w, int zero)
{
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;

        const int hc0 = h_offset * dilation_h - pad_h;
        const int wc0 = w_offset * dilation_w - pad_w;
        for (int h = 0; h < height_col; ++h) {
            int h_pad = h * stride_h + hc0;

            const int row_offset = (c * height_col + h) * width_col;
            const int srow_offset = (c_im * height + h_pad) * width;
            for (int w = 0; w < width_col; ++w) {
                int w_pad = w * stride_w + wc0;
                if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width)))
                    data_col[row_offset + w] = data_im[srow_offset + w_pad];
                else {
                    data_col[row_offset + w] = zero;
                }
            }
        }
    }
}

void qim2row(uint8 * data_col, uint8 * data_im, int height, int width, int channels_col,
             int height_col, int width_col, int kernel_h, int kernel_w, int stride_h, int stride_w,
             int pad_h, int pad_w, int dilation_h, int dilation_w, int zero)
{
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;

        const int hc0 = h_offset * dilation_h - pad_h;
        const int wc0 = w_offset * dilation_w - pad_w; 
        for (int h = 0; h < height_col; ++h) { 
            int h_pad = h * stride_h + hc0;

            const int row_offset = h * width_col;
            const int srow_offset = (c_im * height + h_pad) * width;
            for (int w = 0; w < width_col; ++w) {
                int w_pad = w * stride_w + wc0;
                if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width)))
                    data_col[c + (row_offset + w) * channels_col] = data_im[srow_offset + w_pad];
                else {
                    data_col[c + (row_offset + w) * channels_col] = zero;
                }
            }
        }
    }
}

void qconv2d(uint8 * y, uint8 * x, uint8 * x_col, uint8 * weight, int * bias, int xn, int xc, int xh, int xw, 
            int ko, int ki, int kh, int kw, int ph, int pw, int sh, int sw, int dh, int dw,
            int zero_x, int zero_w, int zero_b, int zero_y, Fixed_point * coe, int rshift, int qmin, int qmax)
{
    int kernel_h = kh;
    int kernel_w = kw;
    int dilation_h = dh;
    int dilation_w = dw;
    int height = xh;
    int width = xw;
    int pad_h = ph;
    int pad_w = pw;
    int stride_h = sh;
    int stride_w = sw;
    int channels = xc;

    int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
    int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
    int height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;

    int yh = height_col;
    int yw= width_col;

    qim2col(x_col, x, xh, xw, channels_col, height_col, width_col, kh, kw, sh, sw, ph, pw, dh, dw, zero_x);
   
    // M: ko, N: yh*yw, K: ki*kh*kw
    // Fixed_point * fp_temp = Fixed_point_init(0);
    int M = ko;
    int N = yh*yw;
    int K = ki*kh*kw;
    #pragma omp parallel for
    for(int i = 0; i<M; i++) {
        Fixed_point * fp_temp = Fixed_point_init(0);
        for(int j = 0; j<N; j++) {
            int temp = 0;
            for(int k = 0; k<K; k++) {
                temp += ((int)weight[i*K + k] - zero_w) * 
                    ((int)x_col[k*N + j] - zero_x);
            }
            temp += bias[i] - zero_b;
            fp_temp->assign(fp_temp, temp);
            fp_temp->mult(fp_temp, coe);
            int t = fp_temp->to_int(fp_temp);
            temp = (t >> rshift) + zero_y;
            y[i*N + j] = (uint8)(
                (temp < 0) ? 0:
                (temp > 255) ? 255:
                temp
            );
        }
        fp_temp->free(fp_temp);
    }

    // fp_temp->free(fp_temp);
}

void qrelu(uint8 * y, uint8 * x, int len, int zero)
{
    for(int i = 0; i<len; i++) {
        y[i] = (x[i] > zero) ? x[i] : zero;
    }
}

void qmaxpool2d(uint8 * y, uint8 * x, int xn, int xc, int xh, int xw, int kh, int kw, int sh, int sw, 
               int ph, int pw, int dh, int dw, int zero)
{
    int padded_h = xh + ph*2;
    int padded_w = xw + pw*2;
    uint8 * padded = (uint8*)malloc(sizeof(uint8)*xn*xc*padded_h*padded_w);
    qpadding(padded, x, xn, xc, xh, xw, ph, pw, zero);
    int yh = (padded_h - (dh*(kh-1)+1)) / sh+1;
    int yw = (padded_w - (dw*(kw-1)+1)) / sw+1;
    for(int n = 0; n<xn; n++) {
        for(int c = 0; c<xc; c++) {
            for(int h = 0; h<yh; h++) {
                for(int w = 0; w<yw; w++) {
                    int start_h = h * sh;
                    int start_w = w * sw;
                    int start_kh = 0;
                    int start_kw = 0;
                    uint8 max = padded[
                        n * xc * padded_h * padded_w + 
                        c * padded_h * padded_w + 
                        start_h * padded_w + 
                        start_w];
                    for(int lkh = 0; lkh < kh; lkh++, start_kh += dh) {
                        start_kw = 0;
                        for(int lkw = 0; lkw < kw; lkw++, start_kw += dw) {
                            if(padded[
                                n * xc * padded_h * padded_w + 
                                c * padded_h * padded_w + 
                                (start_h + start_kh) * padded_w + 
                                (start_w + start_kw)] > max) {
                                max = padded[
                                    n * xc * padded_h * padded_w + 
                                    c * padded_h * padded_w + 
                                    (start_h + start_kh) * padded_w + 
                                    (start_w + start_kw)];
                            }
                        }
                    }
                    y[
                        n * xc * yh * yw + 
                        c * yh * yw + 
                        h * yw + 
                        w
                    ] = max;
                }
            }
        }
    }
    free(padded);
}

void qavgpool2d(uint8 * y, uint8 * x, int xn, int xc, int xh, int xw, int kh, int kw, int sh, int sw, 
               int ph, int pw, int zero)
{
    int padded_h = xh + ph*2;
    int padded_w = xw + pw*2;
    uint8 * padded = (uint8*)malloc(sizeof(uint8)*xn*xc*padded_h*padded_w);
    qpadding(padded, x, xn, xc, xh, xw, ph, pw, zero);
    int yh = (padded_h - kh) / sh+1;
    int yw = (padded_w - kw) / sw+1;
    for(int n = 0; n<xn; n++) {
        for(int c = 0; c<xc; c++) {
            for(int h = 0; h<yh; h++) {
                for(int w = 0; w<yw; w++) {
                    int start_h = h * sh;
                    int start_w = w * sw;
                    int start_kh = 0;
                    int start_kw = 0;
                    int sum = 0;
                    for(int lkh = 0; lkh < kh; lkh++, start_kh ++) {
                        start_kw = 0;
                        for(int lkw = 0; lkw < kw; lkw++, start_kw ++) {
                            sum += padded[
                                n * xc * padded_h * padded_w + 
                                c * padded_h * padded_w + 
                                (start_h + start_kh) * padded_w + 
                                (start_w + start_kw)];
                        }
                    }
                    y[
                        n * xc * yh * yw + 
                        c * yh * yw + 
                        h * yw + 
                        w
                    ] = (uint8)(sum / (kh * kw));
                }
            }
        }
    }
    free(padded);
}

void qflatten(uint8 * y, uint8 * x, int len)
{
    memcpy(y, x, sizeof(uint8)*len);
}

void qfc(uint8 * y, uint8 * x, uint8 * weight, int * bias, int oc, int ic,
        int zero_x, int zero_w, int zero_b, int zero_y, Fixed_point * coe, int rshift, int qmin, int qmax)
{
    // M: 1 N: oc K: ic
    Fixed_point * fp_temp = Fixed_point_init(0);
    for(int o = 0; o<oc; o++) {
        int temp = 0;
        for(int i = 0; i<ic; i++) {
            temp += ((int)x[i] - zero_x) *
                ((int)weight[i*oc + o] - zero_w);
        }
        temp += bias[o] - zero_b;
        fp_temp->assign(fp_temp, temp);
        fp_temp->mult(fp_temp, coe);
        int t = fp_temp->to_int(fp_temp);
        temp = (t >> rshift) + zero_y;
        y[o] = (uint8)(
            (temp < 0) ? 0 :
            (temp > 255) ? 255 : 
            temp
        );
    }
    
    fp_temp->free(fp_temp);
}

void qdropout(uint8 * y, uint8 * x, int len)
{
    memcpy(y, x, sizeof(uint8)*len);
}

void qoutput(uint8 * y, uint8 * x, int len)
{
    memcpy(y, x, sizeof(uint8)*len);
}

void qadd(uint8 * y, uint8 * x1, uint8 * x2, int len, int zero_x1, int zero_x2, int zero_y,
        Fixed_point * coe1, Fixed_point * coe2, int rshift1, int rshift2, int qmin, int qmax)
{
    int * temp_x1 = (int*)malloc(sizeof(int)*len);
    int * temp_x2 = (int*)malloc(sizeof(int)*len);
    Fixed_point * fp_temp1 = Fixed_point_init(0);
    Fixed_point * fp_temp2 = Fixed_point_init(0);
    for(int i = 0; i<len; i++) {
        int temp1 = (int)x1[i] - zero_x1;
        fp_temp1->assign(fp_temp1, temp1);
        fp_temp1->mult(fp_temp1, coe1);
        int t1 = fp_temp1->to_int(fp_temp1);
        if(rshift1 < 0) {
            temp_x1[i] = t1 << (-rshift1);
        }
        else {
            temp_x1[i] = t1 >> rshift1;
        }
    }
    for(int i = 0; i<len; i++) {
        int temp2 = (int)x2[i] - zero_x2;
        fp_temp2->assign(fp_temp2, temp2);
        fp_temp2->mult(fp_temp2, coe2);
        int t2 = fp_temp2->to_int(fp_temp2);
        if(rshift2 < 0) {
            temp_x2[i] = t2 << (-rshift2);
        }
        else {
            temp_x2[i] = t2 >> rshift2;
        }
    }
    for(int i = 0; i<len; i++) {
        int temp = temp_x1[i] + temp_x2[i] + zero_y;
        y[i] = (uint8)(
            (temp < 0) ? 0 : 
            (temp > 255) ? 255 : 
            temp
        );
    }

    free(temp_x1);
    free(temp_x2);
    fp_temp1->free(fp_temp1);
    fp_temp2->free(fp_temp2);
}

void igemm(int * C, uint8 * A, uint8 * B, int M, int N, int K, int accumulate, int zero_A, int zero_B)
{
    #pragma omp parallel for
    for(int i = 0; i<M; i++) {
        for(int j = 0; j<N; j++) {
            int temp = 0;
            for(int k = 0; k<K; k++) {
                temp += ((int)A[i*K + k] - zero_A) * 
                    ((int)B[k*N + j] - zero_B);
            }
            C[i*N + j] = C[i*N + j] * accumulate + temp;
        }
    }
}

int argmax(uint8 * x, int len)
{
    uint8 temp = x[0];
    int index = 0;
    for(int i = 0; i<len; i++) {
        if(x[i] > temp) {
            temp = x[i];
            index = i;
        }
    }
    return index;
}
"""
    return code



def save_weight(
    calculation_graph
):
    '''
    save weight files into target directory
    '''
    if(not os.path.exists("output/fpga_deploy/weight")):
        os.system("mkdir -p output/fpga_deploy/weight")
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
            type(node) == op.QDense):
            weight_path = "output/fpga_deploy/weight/layer_%d_weight" \
                ".bin"%(n)
            bias_path = "output/fpga_deploy/weight/layer_%d_bias.bin"%(n)
            node.weight.tofile(weight_path)
            node.bias.tofile(bias_path)



def gen_cmake():
    '''
    Generate CMakelists.txt
    '''
    code = ""
    code += """
cmake_minimum_required(VERSION 3.5)
project(int_deploy)
set(CMAKE_C_FLAGS "-O3 -Wall -W -pthread -fopenmp")
set(CMAKE_BUILD_TYPE "Release")
include_directories(
    /opt/OpenBLAS/include
)
link_directories(
    /opt/OpenBLAS/lib
)
add_executable(int_deploy main.c fixed_point.c call_lib.c)
target_link_libraries(int_deploy openblas pthread m)    
"""
    return code



def gen_main(
    calculation_graph
):
    '''
    Generate main.c to run on arm cpu
    '''
    # count the number of outputs
    result_count = 0
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Output or 
            type(node) == op.QOutput):
            result_count += 1

    code = ""
    code += "#include \"call_lib.h\"\n"
    code += "int main(void)\n"
    code += "{\n"
    indent = "\t"
    # load dataset
    code += indent + "// load your dataset\n"
    code += "\n"
    # init weight and result memory
    code += indent + "// init weight and result memory\n"
    code += indent + "init();\n"
    code += "\n"
    # alloc space for result
    code += indent + "// alloc space for result\n"
    code += indent + "uint8 ** result = (uint8**)malloc(sizeof(uint8*)*%d)" \
        ";\n"%(result_count)
    code += "\n"
    # ??????
    code += indent + "// calculation\n"
    code += indent + "calc(result, /* pointer to test data */);\n"
    code += "}\n"

    return code



def gen_call_lib_h(
    
):
    '''
    Generate call_lib.h to run on arm cpu
    '''
    code = ""
    code += "#include <stdio.h>\n"
    code += "#include <stdlib.h>\n"
    code += "#include <string.h>\n"
    code += "#include <assert.h>\n"
    code += "#include <math.h>\n"
    code += "#include <cblas.h>\n"
    code += "#include <pynq_api.h>\n"
    code += "#include <memory.h>\n"
    code += "#include <unistd.h>\n"
    code += "#include <sys/mman.h>\n"
    code += "#include <sys/types.h>\n"
    code += "#include <sys/stat.h>\n"
    code += "#include <fcntl.h>\n"
    code += "#include <time.h>\n"
    code += "#include <pthread.h>\n"
    code += "#include <sys/time.h>\n"
    code += "#include \"fixed_point.h\"\n"

    code += "typedef unsigned char uint8;\n"
    code += "typedef unsigned long long int uint64;\n"
    code += "void init();\n"
    code += "void calc(uint8 ** result , uint8 * data);\n"
    code += "int argmax(uint8 * x, int len);\n"

    return code



def declare_pointer_for_weight_blocks(
    calculation_graph,
    sub_matrix_size,
    calc_process_with_parallel
):
    '''
    declare pointer for weight blocks
    Attention: Since we need to transfer several blocks together sometimes,
    we declare one dma_shared_memory block for these blocks which transfer
    together
    '''
    code = ""
    # generate for conv2d
    for layer_index, layer in enumerate(calc_process_with_parallel):
        block_index = 0
        for pair_index, pair in enumerate(layer):
            left_line = pair[0]
            right_line = pair[1]
            is_load = False
            for action_index, action in enumerate(left_line):
                if(action[0:4] == "load"):
                    is_load = True
            if(not is_load):
                continue
            node_index = get_node_index(layer_index, calculation_graph)
            code += "PYNQ_SHARED_MEMORY layer_%d_weight_%d_shared;\n"%(
                node_index, block_index
            )
            code += "uint8 * layer_%d_weight_%d_dma;\n"%(
                node_index, block_index
            )
            block_index += 1

    # generate for fc
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.QDense):
            continue
        code += "PYNQ_SHARED_MEMORY layer_%d_weight_shared;\n"%(n)
        code += "uint8 * layer_%d_weight_dma;\n"%(n)

    code += "\n\n"

    return code



def alloc_space_for_weight_blocks(
    calculation_graph,
    sub_matrix_size,
    calc_process_with_parallel
):
    '''
    allocate space for weight blocks
    '''
    code = ""
    # generate for conv2d
    for layer_index, layer in enumerate(calc_process_with_parallel):
        block_index = 0
        for pair_index, pair in enumerate(layer):
            left_line = pair[0]
            right_line = pair[1]
            is_load = False
            for action_index, action in enumerate(left_line):
                if(action[0:4] == "load"):
                    is_load = True
            if(not is_load):
                continue
            node_index = get_node_index(layer_index, calculation_graph)
            # calc allocate size
            total_size = 0
            for action_index, action in enumerate(left_line):
                if(not action[0:4] == "load"):
                    continue
                block = action.split(" ")[1]
                matrix = block.split("_")[0]
                row = int(block.split("_")[1])
                col = int(block.split("_")[2])
                if(matrix != "A"):
                    continue
                shape = sub_matrix_size[layer_index][0][row][col]
                total_size += shape[0] * shape[1]
            code += "PYNQ_allocatedSharedMemory(&layer_%d_weight_%d_shared, " \
                "sizeof(uint8)*%d, 1);\n"%(node_index, block_index, total_size)
            code += "layer_%d_weight_%d_dma = (uint8*)layer_%d_weight_%d_" \
                "shared.pointer;\n"%(node_index, block_index, 
                node_index, block_index)
            block_index += 1
    
    # generate for fc
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.QDense):
            continue
        code += "PYNQ_allocatedSharedMemory(&layer_%d_weight_shared, sizeof(" \
            "uint8)*%d, 1);\n"%(n, node.weight.shape[0] * node.weight.shape[1])
        code += "layer_%d_weight_dma = layer_%d_weight_shared.pointer;\n"%(n, n)
    
    code += "\n\n"

    return code



def im2col_weight(
    calculation_graph,
    im2col_shape,
    sub_matrix_size,
    calc_process_with_parallel
):
    '''
    copy weight into weight blocks
    '''
    code = ""
    code += "uint8 * weight_y_ptr;\n"
    code += "uint8 * weight_x_ptr;\n"

    # copy conv weights
    for layer_index, layer in enumerate(calc_process_with_parallel):
        block_index = 0
        for pair_index, pair in enumerate(layer):
            left_line = pair[0]
            right_line = pair[1]
            is_load = False
            for action_index, action in enumerate(left_line):
                if(action[0:4] == "load"):
                    is_load = True
            if(not is_load):
                continue
            node_index = get_node_index(layer_index, calculation_graph)
            # get dma block shape
            #! Do I need to get dma block shape? No! I just need to copy 
            #! into it contiguously. I just need to record how many bytes
            #! I have copied.
            copied_size = 0
            for action_index, action in enumerate(left_line):
                # copy by im2col block
                if(not action[0:4] == "load"):
                    continue
                block = action.split(" ")[1]
                matrix = block.split("_")[0]
                row = int(block.split("_")[1])
                col = int(block.split("_")[2])
                if(matrix != "A"):
                    continue
                # get submatrix block shape
                block_shape = sub_matrix_size[layer_index][0][row][col]
                # get original im2col block shape(before fit to 2^n)
                # # calc cut left_up coordinate according to submatrix_size
                cut_coordinate_row = 0
                cut_coordinate_col = 0
                for i in range(row):
                    cut_coordinate_row += sub_matrix_size[layer_index] \
                        [0][i][col][0]
                for i in range(col):
                    cut_coordinate_col += sub_matrix_size[layer_index] \
                        [0][row][i][1]
                # # calc cut right_down coordinate according to submatrix_size
                cut_coordinate_row_upper = cut_coordinate_row + \
                    sub_matrix_size[layer_index][0][row][col][0]
                cut_coordinate_col_upper = cut_coordinate_col + \
                    sub_matrix_size[layer_index][0][row][col][1]
                # # get im2col border 
                row_border = im2col_shape[layer_index][0][0]
                col_border = im2col_shape[layer_index][0][1]
                # # calc exceed
                row_exceed = max(cut_coordinate_row_upper - row_border, 0)
                col_exceed = max(cut_coordinate_col_upper - col_border, 0)
                # # original im2col block shape
                im2col_block_shape = (block_shape[0] - row_exceed,
                    block_shape[1] - col_exceed)
                
                # copy from `im2col_block` to `block`
                dst_start = copied_size
                src_start = cut_coordinate_row * col_border + cut_coordinate_col
                code += '''
// copy layer_%d: %s
weight_y_ptr = &layer_%d_weight_%d_dma[%d];
weight_x_ptr = &layer_%d_weight[%d];
for(int h = 0; h<%d; h++) {
    memcpy(weight_y_ptr, weight_x_ptr, sizeof(uint8)*%d);
    weight_y_ptr += %d;
    weight_x_ptr += %d;
'''%(
                    node_index, block,
                    node_index, block_index, dst_start, 
                    node_index, src_start,
                    im2col_block_shape[0],
                    im2col_block_shape[1],
                    im2col_block_shape[1],
                    col_border,
                )
                if(col_exceed > 0):
                    code += '''
    for(int t = 0; t<%d; t++) {
        weight_y_ptr[t] = %d;
    }
    weight_y_ptr += %d;
'''%(
                    col_exceed,
                    calculation_graph[node_index].zero_w,
                    col_exceed,
                )
                code += "}\n"
                if(row_exceed > 0):
                    code += '''
for(int h = 0; h<%d; h++) {
    memset(weight_y_ptr, %d, sizeof(uint8)*%d);
    weight_y_ptr += %d;
}
'''%(
                    row_exceed,
                    calculation_graph[node_index].zero_w, block_shape[1],
                    block_shape[1],
                )   

                copied_size += block_shape[0] * block_shape[1]
            block_index += 1
    

    # copy fc weights
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.QDense):
            continue
        code += '''
// copy layer %d: transepose fc weight
for(int oc = 0; oc < %d; oc++) {
    for(int ic = 0; ic < %d; ic++) {
        layer_%d_weight_dma[oc*%d+ic] = layer_%d_weight[ic*%d+oc];
    }
}
'''%(
            n, 
            node.output_channel,
            node.input_channel,
            n, node.input_channel, n, node.output_channel
        )   


    # free original weights
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
            type(node) == op.QDense):
            code += "free(layer_%d_weight);\n"%(n)
            code += "free(layer_%d_bias);\n"%(n)


    return code



def declare_pointer_for_medium_results(
    calculation_graph,
    im2col_shape,
    sub_matrix_size,
    calc_process_with_parallel
):
    '''
    declare pointer for medium results
    '''
    code = ""
    for n, node in enumerate(calculation_graph):
        code += "uint8 * layer_%d_res;\n"%(n)
        if(type(node) == op.QConv2d):
            # for im2col result, declare contiguous space
            conv_index = get_conv2d_index(n, calculation_graph)
            xcol_shape = im2col_shape[conv_index][1]
            code += "uint8 * layer_%d_xcol;\n"%(n)
            # for im2col_dma, declare for each transfer
            calc_process = calc_process_with_parallel[conv_index]
            block_index = 0
            for pair_index, pair in enumerate(calc_process):
                left_line = pair[0]
                right_line = pair[1]
                is_load = False
                for action_index, action in enumerate(left_line):
                    if(action[0:4] == "load"):
                        is_load = True
                if(not is_load):
                    continue
                code += "PYNQ_SHARED_MEMORY layer_%d_xcol_%d_shared;\n"%(
                    n, block_index)
                code += "uint8 * layer_%d_xcol_%d_dma;\n"%(n, block_index)
                block_index += 1
            # for matmul_res, declare for each transfer
            block_index = 0
            for pair_index, pair in enumerate(calc_process):
                left_line = pair[0]
                right_line = pair[1]
                is_store = False
                for action_index, action in enumerate(left_line):
                    if(action[0:5] == "store"):
                        is_store = True
                if(not is_store):
                    continue
                code += "PYNQ_SHARED_MEMORY layer_%d_matmul_%d_shared;\n"%(
                    n, block_index)
                code += "uint8 * layer_%d_matmul_%d_dma;\n"%(n, block_index)
                block_index += 1
        elif(type(node) == op.QDense):
            code += "PYNQ_SHARED_MEMORY layer_%d_in_shared;\n"%(n)
            code += "uint8 * layer_%d_in_dma;\n"%(n)
            code += "PYNQ_SHARED_MEMORY layer_%d_out_shared;\n"%(n)
            code += "uint64 * layer_%d_out_dma;\n"%(n)
        elif(type(node) == op.QAdd):
            code += "PYNQ_SHARED_MEMORY layer_%d_in1_shared;\n"%(n)
            code += "uint8 * layer_%d_in1_dma;\n"%(n)
            code += "PYNQ_SHARED_MEMORY layer_%d_in2_shared;\n"%(n)
            code += "uint8 * layer_%d_in2_dma;\n"%(n)
            code += "PYNQ_SHARED_MEMORY layer_%d_out_shared;\n"%(n)
            code += "uint8 * layer_%d_out_dma;\n"%(n)
    code += "\n\n"

    return code



def alloc_space_for_medium_results(
    calculation_graph,
    im2col_shape,
    sub_matrix_size,
    calc_process_with_parallel
):
    '''
    allocate space for medium results
    '''
    code = ""
    for n, node in enumerate(calculation_graph):
        res_size = 1
        for s in node.output_shape:
            res_size *= s
        code += "layer_%d_res = (uint8*)malloc(sizeof(uint8)*%d);\n"%(
            n, res_size)
        if(type(node) == op.QConv2d):
            # for im2col result, declare contiguous space
            conv_index = get_conv2d_index(n, calculation_graph)
            xcol_shape = im2col_shape[conv_index][1]
            code += "layer_%d_xcol = (uint8*)malloc(sizeof(uint8)*%d);\n"%(
                n, xcol_shape[0] * xcol_shape[1])
            # for im2col_dma, declare for each transfer
            calc_process = calc_process_with_parallel[conv_index]
            block_index = 0
            for pair_index, pair in enumerate(calc_process):
                left_line = pair[0]
                right_line = pair[1]
                is_load = False
                for action_index, action in enumerate(left_line):
                    if(action[0:4] == "load"):
                        is_load = True
                if(not is_load):
                    continue
                # calc allocate size
                total_size = 0
                for action_index, action in enumerate(left_line):
                    if(not action[0:4] == "load"):
                        continue
                    block = action.split(" ")[1]
                    matrix = block.split("_")[0]
                    row = int(block.split("_")[1])
                    col = int(block.split("_")[2])
                    if(matrix != "B"):
                        continue
                    shape = sub_matrix_size[conv_index][1][row][col]
                    total_size += shape[0] * shape[1]
                code += "PYNQ_allocatedSharedMemory(&layer_%d_xcol_%d_shared" \
                    ", sizeof(uint8)*%d, 1);\n"%(n, block_index, total_size)
                code += "layer_%d_xcol_%d_dma = (uint8*)layer_%d_xcol_%d_" \
                    "shared.pointer;\n"%(n, block_index, n, block_index)
                block_index += 1
            # for matmul_res, declare for each transfer
            block_index = 0
            for pair_index, pair in enumerate(calc_process):
                left_line = pair[0]
                right_line = pair[1]
                is_store = False
                for action_index, action in enumerate(left_line):
                    if(action[0:5] == "store"):
                        is_store = True
                if(not is_store):
                    continue
                total_size = 0
                for action_index, action in enumerate(left_line):
                    if(not action[0:5] == "store"):
                        continue
                    block = action.split(" ")[1]
                    matrix = block.split("_")[0]
                    row = int(block.split("_")[1])
                    col = int(block.split("_")[2])
                    shape = (sub_matrix_size[conv_index][0][row][0][0], 
                        sub_matrix_size[conv_index][1][0][col][1])
                    total_size += shape[0] * shape[1]
                code += "PYNQ_allocatedSharedMemory(&layer_%d_matmul_%d_" \
                    "shared, sizeof(uint8)*%d, 1);\n"%(n, block_index, 
                    total_size)
                code += "layer_%d_matmul_%d_dma = (uint8*)layer_%d_matmul_" \
                    "%d_shared.pointer;\n"%(n, block_index, n, block_index)
                block_index += 1
        elif(type(node) == op.QDense):
            code += "PYNQ_allocatedSharedMemory(&layer_%d_in_shared, sizeof" \
                "(uint8)*%d, 1);\n"%(n, node.input_channel)
            code += "layer_%d_in_dma = (uint8*)layer_%d_in_shared.pointer;\n"%(
                n, n)
            code += "PYNQ_allocatedSharedMemory(&layer_%d_out_shared, " \
                "sizeof(uint64)*%d, 1);\n"%(n, node.output_channel)
            code += "layer_%d_out_dma = (uint64*)layer_%d_out_shared.pointer;" \
                "\n"%(n, n)
        elif(type(node) == op.QAdd):
            total_size = 1
            for s in node.output_shape:
                total_size *= s
            code += "PYNQ_allocatedSharedMemory(&layer_%d_in1_shared, " \
                "sizeof(uint8)*%d, 1);\n"%(n, s)
            code += "layer_%d_in1_dma = (uint8*)layer_%d_in1_shared.pointer" \
                ";\n"%(n, n)
            code += "PYNQ_allocatedSharedMemory(&layer_%d_in2_shared, " \
                "sizeof(uint8)*%d, 1);\n"%(n, s)
            code += "layer_%d_in2_dma = (uint8*)layer_%d_in2_shared.pointer" \
                ";\n"%(n, n)
            code += "PYNQ_allocatedSharedMemory(&layer_%d_out_shared, " \
                "sizeof(uint8)*%d, 1);\n"%(n, s)
            code += "layer_%d_out_dma = (uint8*)layer_%d_out_shared.pointer" \
                ";\n"%(n, n)
    code += "\n\n"

    return code



def declare_bus(
    BRAM_CTRL_0,
    AXI_DMA_R0,
    AXI_DMA_R1,
    AXI_DMA_W0,
):
    '''
    declare dma and bram
    '''

    code = ""

    # declare address
    code += "#define BRAM_CTRL_0 %s\n"%(BRAM_CTRL_0)
    code += "#define AXI_DMA_R0 %s\n"%(AXI_DMA_R0)
    code += "#define AXI_DMA_R1 %s\n"%(AXI_DMA_R1)
    code += "#define AXI_DMA_W0 %s\n"%(AXI_DMA_W0)

    code += "PYNQ_AXI_DMA dma_r0;\n"
    code += "PYNQ_AXI_DMA dma_r1;\n"
    code += "PYNQ_AXI_DMA dma_w0;\n"
    code += "volatile unsigned int * mmio;\n"
    code += "\n\n"

    return code



def open_bus():
    '''
    open dma and bram
    '''
    code = ""

    # open dma
    code += '''
PYNQ_openDMA(&dma_r0, AXI_DMA_R0);
PYNQ_openDMA(&dma_r1, AXI_DMA_R1);
PYNQ_openDMA(&dma_w0, AXI_DMA_W0);    
printf("open dma finished\\n");
'''

    # open bram
    code += '''
int fd = open("/dev/mem", O_RDWR|O_SYNC);
if(fd == -1) {
    printf("open /dev/mem error\\n"); 
    exit(-1);
}
mmio = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_CTRL_0);
if(mmio == 0) {
    printf("mmap failed\\n"); 
    exit(-1);
}
printf("open bram finished\\n");     
'''

    return code



def generate_init(
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel
):
    '''
    Generate init() function of call_lib
    '''
    code = ""

    code += "void init()\n"
    code += "{\n"

    # alloc space for weights
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "layer_%d_weight = (uint8*)malloc(sizeof(uint8)*%d);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "layer_%d_bias = (int*)malloc(sizeof(int)*%d);\n"%(
                n, node.bias.shape[0])
        elif(type(node) == op.QDense):
            code += "layer_%d_weight = (uint8*)malloc(sizeof(uint8)*%d);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1])
            code += "layer_%d_bias = (int*)malloc(sizeof(int)*%d);\n"%(
                n, node.bias.shape[0])
    code += "\n\n"

    # read weights into memory
    code += "FILE * f;\n"
    code += "int l;\n"
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "f = fopen(\"../weight/layer_%d_weight.bin\", \"rb\");\n"%(
                n)
            code += "l = fread(layer_%d_weight, sizeof(uint8), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "fclose(f);\n"
            code += "f = fopen(\"../weight/layer_%d_bias.bin\", \"rb\");\n"%(n)
            code += "l = fread(layer_%d_bias, sizeof(int), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
        elif(type(node) == op.QDense):
            code += "f = fopen(\"../weight/layer_%d_weight.bin\", \"rb\");\n"%(
                n)
            code += "l = fread(layer_%d_weight, sizeof(uint8), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1])
            code += "fclose(f);\n"
            code += "f = fopen(\"../weight/layer_%d_bias.bin\", \"rb\");\n"%(n)
            code += "l = fread(layer_%d_bias, sizeof(int), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
    
    # alloc space for weight blocks
    code += alloc_space_for_weight_blocks(
        calculation_graph,
        sub_matrix_size,
        calc_process_with_parallel
    )

    # alloc space for medium results
    code += alloc_space_for_medium_results(
        calculation_graph,
        im2col_shape,
        sub_matrix_size,
        calc_process_with_parallel
    )

    # copy weight into block weights
    code += im2col_weight(
        calculation_graph,
        im2col_shape,
        sub_matrix_size,
        calc_process_with_parallel
    )

    # open dma and bram
    code += open_bus()


    code += "}\n"
    code += "\n\n"

    return code



def gen_QInput(n, calculation_graph):
    '''
    generate Input
    '''
    code = ""
    node = calculation_graph[n]
    length = 1
    for i in node.output_shape:
        length *= i
    code += "// layer %d: %s\n"%(n, node)
    code += "qinput(layer_%d_res, data, %d);\n"%(
        n, length
    )
    code += "\n\n"

    return code



def gen_copy(
    n, # node index
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel,
    pair_index,
    part
):
    '''
    generate code for copy action
    How to copy B?
    Now we use im2row to get a cache friendly im2col result.
    For im2row result, the left side length is `height_col * width*col`,
    the up side length is `channel * kernel_h * kernel_w`,
    in other words, it is the transpose of im2col result.

    Since we designed in im2col, but use im2row actually, so some data
    should be process specially
    '''

    def get_block_index(
        n,
        calculation_graph,
        calc_process_with_parallel,
        current_pair_index,
        part,
        type
    ):
        '''
        Get dma shared block index(this function should be called by gen_copy)
        '''
        conv_index = get_conv2d_index(n, calculation_graph)
        calc_process = calc_process_with_parallel[conv_index]
        action_group = calc_process[current_pair_index][part]
        matrices = []
        for action in action_group:
            target = action.split(" ")[1]
            matrices.append(target)

        if(type == "load"):
            block_index = 0
            for pair_index, pair in enumerate(calc_process):
                left_line = pair[0]
                right_line = pair[1]
                if(pair_index < current_pair_index):
                    continue
                is_load = False
                for action_index, action in enumerate(left_line):
                    if(action[0:4] == "load"):
                        is_load = True
                if(not is_load):
                    continue
                matrices_in_load = []
                for action_index, action in enumerate(left_line):
                    if(action[0:4] != "load"):
                        continue
                    target = action.split(" ")[1]
                    matrices_in_load.append(target)
                if(matrices == matrices_in_load):
                    return block_index
                block_index += 1
        elif(type == "store"):
            block_index = 0
            pair_index = current_pair_index
            while(pair_index >= 0):
                pair = calc_process[pair_index]
                pair_index -= 1
                left_line = pair[0]
                right_line = pair[1]
                is_store = False
                for action_index, action in enumerate(left_line):
                    if(action[0:5] == "store"):
                        is_store = True
                if(not is_store):
                    continue
                matrices_in_store = []
                for action_index, action in enumerate(left_line):
                    if(action[0:5] != "store"):
                        continue
                    target = action.split(" ")[1]
                    matrices_in_store.append(target)
                if(matrices == matrices_in_store):
                    return block_index
                block_index += 1

    code = ""
    
    node = calculation_graph[n]
    conv_index = get_conv2d_index(n, calculation_graph)
    
    calc_process = calc_process_with_parallel[conv_index]
    calc_process_pair = calc_process[pair_index]
    action_group = calc_process_pair[part]
    submatrix_size_A = sub_matrix_size[conv_index][0]
    submatrix_size_B = sub_matrix_size[conv_index][1]
    # calc submatrix_size_C
    block_count_row_A = len(submatrix_size_A)
    block_count_col_A = len(submatrix_size_A[0])
    block_count_row_B = len(submatrix_size_B)
    block_count_col_B = len(submatrix_size_B[0])
    submatrix_size_C = []
    for row in range(block_count_row_A):
        submatrix_size_C.append([])
        for col in range(block_count_col_B):
            submatrix_size_C[row].append((
                submatrix_size_A[row][0][0],
                submatrix_size_B[0][col][1]
            ))
    
    copied_size = 0 # the size of data which has copied into dma_shared_memory
    for action_index, action in enumerate(action_group):
        block = action.split(" ")[1]
        matrix = block.split("_")[0]
        row = int(block.split("_")[1])
        col = int(block.split("_")[2])
        
        if(matrix == "A"):
            # weight is copied in init()
            continue
        elif(matrix == "B"):
            # copy feature map now
            # 1. find the left up point of the block in im2col matrix
            start_row = 0
            for r in range(row):
                start_row += submatrix_size_B[r][col][0]
            start_col = 0
            for c in range(col):
                start_col += submatrix_size_B[row][c][1]
            # 2. find the right down point of the block in im2col matrix
            # (fit to 2^n)
            end_row = start_row + submatrix_size_B[row][col][0]
            end_col = start_col + submatrix_size_B[row][col][1]
            # 3. find the true right down point(without fit to 2^n)
            row_upper_bound = im2col_shape[conv_index][1][0]
            col_upper_bound = im2col_shape[conv_index][1][1]
            end_row_real = min(end_row, row_upper_bound)
            end_col_real = min(end_col, col_upper_bound)
            # 4. calc the exceed length
            row_exceed = max(end_row - end_row_real, 0)
            col_exceed = max(end_col - end_col_real, 0)
            # 5. get start_coordinate in im2row result
            start_coordinate = (start_col, start_row)
            # 6. get start_address in dma_shared_memory
            start_address_dma = copied_size
            # 7. update copied size
            current_copy_size = submatrix_size_B[row][col][0] * \
                submatrix_size_B[row][col][1]
            copied_size += current_copy_size
            # 8. get im2row shape
            im2row_shape = (im2col_shape[conv_index][1][1], 
                im2col_shape[conv_index][1][0])
            # 9. get start_address in im2row result
            start_address_xcol = start_coordinate[0] * im2row_shape[1] + \
                start_coordinate[1]
            # 10. get actually copy control data in im2row(not in im2col now)
            copy_rows = end_col_real - start_col
            copy_data_each_row = end_row_real - start_row
            dma_increament = submatrix_size_B[row][col][0]
            xcol_increament = im2row_shape[1]
            # 11. get dma shared block number
            block_index = get_block_index(
                n,
                calculation_graph,
                calc_process_with_parallel,
                pair_index,
                part,
                "load"
            )
            # 12. generate copy code
            code += """
// // copy %s
dst_ptr_uint8 = &layer_%d_xcol_%d_dma[%d];
src_ptr_uint8 = &layer_%d_xcol[%d];
for(int h = 0; h<%d; h++) {
    memcpy(dst_ptr_uint8, src_ptr_uint8, sizeof(uint8)*%d);
    dst_ptr_uint8 += %d;
    src_ptr_uint8 += %d;
}
            \n"""%(
                block, 
                n, block_index, start_address_dma,
                n, start_address_xcol,
                copy_rows,
                copy_data_each_row,
                dma_increament,
                xcol_increament
            )
            code += "\n"
        elif(matrix == "C"):
            # copy result now
            # 1. find the left up point of the block in im2col matrix
            start_row = 0
            for r in range(row):
                start_row += submatrix_size_C[r][col][0]
            start_col = 0
            for c in range(col):
                start_col += submatrix_size_C[row][c][1]
            # 2. find the right down point of the block in im2col matrix
            # (fit to 2^n)
            end_row = start_row + submatrix_size_C[row][col][0]
            end_col = start_col + submatrix_size_C[row][col][1]
            # 3. find the true right down point(without fit to 2^n)
            row_upper_bound = im2col_shape[conv_index][0][0]
            col_upper_bound = im2col_shape[conv_index][1][1]
            end_row_real = min(end_row, row_upper_bound)
            end_col_real = min(end_col, col_upper_bound)
            # 4. calc the exceed length
            row_exceed = max(end_row - end_row_real, 0)
            col_exceed = max(end_col - end_col_real, 0)
            # 5. get start_coordinate in result matrix
            start_coordinate = (start_row, start_col)
            # 6. get start_address in dma_shared_memory
            start_address_dma = copied_size
            # 7. update copied size
            current_copy_size = submatrix_size_C[row][col][0] * \
                submatrix_size_C[row][col][1]
            copied_size += current_copy_size
            # 8. get result_matrix shape
            result_shape = (im2col_shape[conv_index][0][0], 
                im2col_shape[conv_index][1][1])
            # 9. get start_address in result matrix
            start_address_result = start_coordinate[0] * result_shape[1] + \
                start_coordinate[1]
            # 10. get actually copy control data in matmul_res
            copy_rows = end_row_real - start_row
            copy_data_each_row = end_col_real - start_col
            dma_increament = submatrix_size_C[row][col][1]
            result_increament = result_shape[1]
            # 11. get dma shared block number
            block_index = get_block_index(
                n,
                calculation_graph,
                calc_process_with_parallel,
                pair_index,
                part,
                "store"
            )
            # 12. generate copy code
            code += """
// // copy %s
dst_ptr_uint8 = &layer_%d_res[%d];
src_ptr_uint8 = &layer_%d_matmul_%d_dma[%d];
for(int h = 0; h < %d; h++) {
    memcpy(dst_ptr_uint8, src_ptr_uint8, sizeof(uint8)*%d);
    dst_ptr_uint8 += %d;
    src_ptr_uint8 += %d;
}
            \n"""%(
                block,
                n, start_address_result,
                n, block_index, start_address_dma,
                copy_rows,
                copy_data_each_row,
                result_increament,
                dma_increament
            )
        else:
            raise ValueError("Unknown matrix block")

    return code



def gen_set_dma(
    n, 
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel,
    instr_analyse_result,
    resource_analyse_result,
    instruction_index,
    pair_index,
    part
):
    '''
    Generate code for set_dma action
    '''
    code = ""

    def get_block_index(
        n,
        calculation_graph,
        calc_process_with_parallel,
        current_pair_index,
        part,
        type
    ):
        '''
        Get dma shared block index(this should be called by set_dma)
        '''
        node = calculation_graph[n]
        conv_index = get_conv2d_index(n, calculation_graph)
        calc_process = calc_process_with_parallel[conv_index]
        calc_process_pair = calc_process[current_pair_index]
        action_group = calc_process_pair[part]
        
        block_index = 0
        if(type == "load"):
            for pair_index, pair in enumerate(calc_process):
                if(pair_index >= current_pair_index):
                    break
                left_line = pair[0]
                right_line = pair[1]
                if(left_line[0][0:4] == "load"):
                    block_index += 1
            return block_index
        elif(type == "store"):
            for pair_index, pair in enumerate(calc_process):
                if(pair_index >= current_pair_index):
                    break
                left_line = pair[0]
                right_line = pair[1]
                if(left_line[0][0:5] == "store"):
                    block_index += 1
            return block_index
        else:
            raise TypeError("Unknown transmission type")

    
    node = calculation_graph[n]
    conv_index = get_conv2d_index(n, calculation_graph)
    
    calc_process = calc_process_with_parallel[conv_index]
    calc_process_pair = calc_process[pair_index]
    action_group = calc_process_pair[part]

    if(len(action_group) != 1):
        raise ValueError("There should be only 1 set_dma instruction in" \
            "its group")
    
    direction = action_group[0].split(" ")[1]
    
    # Since the matrices need to transfer through dma is recorded in next
    # calc_process_pair, we should read it
    calc_process_pair = calc_process[pair_index + 1]
    # Since the load and calc actions are always in left_line, read it
    action_group = calc_process_pair[0]

    submatrix_size_A = sub_matrix_size[conv_index][0]
    submatrix_size_B = sub_matrix_size[conv_index][1]
    
    if(direction == "load"):
        matrix_A = []   # matrix A's that need to be transferred
        matrix_B = []   # matrix B's that need to be transferred
        for action_index, action in enumerate(action_group):
            if(action[0:4] == "load"):
                target = action.split(" ")[1]
                matrix = target.split("_")[0]
                if(matrix == "A"):
                    matrix_A.append(target)
                elif(matrix == "B"):
                    matrix_B.append(target)
                else:
                    raise TypeError("Unknown matrix name")
        # calc data size of A that need to be transferred
        total_size_A = 0
        for block in matrix_A:
            row = int(block.split("_")[1])
            col = int(block.split("_")[2])
            shape = submatrix_size_A[row][col]
            size = shape[0] * shape[1]
            total_size_A += size
        # calc data size of B that need to be transferred
        total_size_B = 0
        for block in matrix_B:
            row = int(block.split("_")[1])
            col = int(block.split("_")[2])
            shape = submatrix_size_B[row][col]
            size = shape[0] * shape[1]
            total_size_B += size
        # get the amount of data that each line of bram can hold
        max_len_support = resource_analyse_result["max_matrix_len_support"]
        bram_group = resource_analyse_result["bram_group"]
        data_per_line = max_len_support if(bram_group == 0) else \
            max_len_support * bram_group
        # set calculation type(0 for conv)
        calculation_type = 0
        # calc bram lines need by A and B
        bram_line_A_need = total_size_A // data_per_line
        bram_line_B_need = total_size_B // data_per_line
        if(total_size_A % data_per_line != 0 or 
            total_size_B % data_per_line != 0):
            raise ValueError("data should fill a bram line completly")
        # get pl instruction begin and end address
        begin_address = 0
        end_address = 0
        for item in instruction_index:
            if(conv_index == item[0][0] and 
                pair_index+1 == item[0][1]):
                begin_address = item[1]
                end_address = item[2]
                break
        # get ps instruction width
        ps_bit_width_need_for_conv = instr_analyse_result[
            "ps_bit_width_need_for_conv"]
        ps_bit_width_need = instr_analyse_result["ps_bit_width_need"]
        # list instruction fields(each item: (width, value))
        # because of the design of top.v, the end_address write into 
        # ps_instruction should be `end_address+1`
        instruction_fields = [
            (instr_analyse_result["ps_calculation_type"],
                calculation_type),
            (instr_analyse_result["ps_weight_data_length_for_conv"],
                bram_line_A_need),
            (instr_analyse_result["ps_feature_map_data_length_for_conv"],
                bram_line_B_need),
            (instr_analyse_result["ps_instr_begin_addr_for_conv"],
                begin_address),
            (instr_analyse_result["ps_instr_end_addr_for_conv"],
                end_address + 1),
            (ps_bit_width_need - ps_bit_width_need_for_conv, 0)
        ]
        # generate binary instruction code
        instruction_code = ""
        for item in instruction_fields:
            instruction_code += decimal_to_bin(item[1], item[0])
        # generate hexadecimal instruction code
        instruction_code = bin_to_hex(instruction_code)
        # divide into 32bit per word
        words = []
        for i in range(ps_bit_width_need // 32):
            words.append(instruction_code[i*8 : i*8+8])
        # get block index
        block_index = get_block_index(
            n,
            calculation_graph,
            calc_process_with_parallel,
            pair_index,
            part,
            "load"
        )
        # generate code
        code += "// // set_dma\n"
        for count, word in enumerate(words):
            code += "mmio[%d] = 0x%s;\n"%(count, word)
        code += "mmio[%d] = 0xFFFFFFFF;\n"%(count+1)
        code += """
PYNQ_writeDMA(&dma_r0, &layer_%d_weight_%d_shared, 0, sizeof(uint8)*%d);
PYNQ_writeDMA(&dma_r1, &layer_%d_xcol_%d_shared, 0, sizeof(uint8)*%d);
PYNQ_waitForDMAComplete(&dma_r0, AXI_DMA_WRITE);   
PYNQ_waitForDMAComplete(&dma_r1, AXI_DMA_WRITE);
        \n"""%(
            n, block_index, total_size_A,
            n, block_index, total_size_B, 
        )
    elif(direction == "store"):
        matrix_C = []
        for action_index, action in enumerate(action_group):
            if(action[0:5] == "store"):
                target = action.split(" ")[1]
                matrix = target.split("_")[0]
                if(matrix == "C"):
                    matrix_C.append(target)
                else:
                    raise TypeError("Unknown matrix name")
        # calc data size of C that need to be transferred
        total_size_C = 0
        for block in matrix_C:
            row = int(block.split("_")[1])
            col = int(block.split("_")[2])
            shape = (submatrix_size_A[row][0][0], 
                submatrix_size_B[0][col][1])
            size = shape[0] * shape[1]
            total_size_C += size
        # get block index
        block_index = get_block_index(
            n,
            calculation_graph,
            calc_process_with_parallel,
            pair_index,
            part,
            "store"
        )
        # generate code
        code += """
// // set_dma
PYNQ_readDMA(&dma_w0, &layer_%d_matmul_%d_dma, 0, sizeof(uint8)*%d);
PYNQ_waitForDMAComplete(&dma_w0, AXI_DMA_READ);
        \n"""%(
            n, block_index, total_size_C
        )
    else:
        raise ValueError("Unknown set_dma direction")

    return code



def gen_load(
    n, 
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel,
    pair_index,
    part
):
    '''
    Generate code for load and calc action
    Since the load action is actually finished by `set_dma`, 
    and the calc action is done by fpga automatically,
    this part do not need to generate code
    '''
    code = ""


    return code



def gen_store(
    n, 
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel,
    pair_index,
    part
):
    '''
    Generate code for store action
    Since the store action is actually finished by `set_dma`,
    and the calc action is done by fpga automatically,
    this part do not need to generate code
    '''
    code = ""


    return code



def gen_QConv2d(
    n, 
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel,
    instr_analyse_result,
    resource_analyse_result,
    instruction_index,
):
    '''
    generate conv2d
    '''
    code = ""
    node = calculation_graph[n]
    input_id = node.input
    input_node = calculation_graph[input_id]
    code += "// layer %d: %s\n"%(n, node)

    # im2row
    code += "// // im2row\n"
    code += "channels_col = %d * %d * %d;\n"%(
                input_node.output_shape[1],
                node.weight.shape[2], node.weight.shape[3])
    code += "dil_kernel_h = (%d - 1) * %d + 1;\n"%(
                node.weight.shape[2], node.dilation[0])
    code += "dil_kernel_w = (%d - 1) * %d + 1;\n"%(
        node.weight.shape[3], node.dilation[1])
    code += "height_col = (%d + 2 * %d - dil_kernel_h) / %d + 1;\n"%(
        input_node.output_shape[2], 
        node.padding[0], node.stride[0])
    code += "width_col = (%d + 2 * %d - dil_kernel_w) / %d + 1;\n"%(
        input_node.output_shape[3],
        node.padding[1], node.stride[1])
    code += "qim2row(layer_%d_xcol, layer_%d_res, %d, %d, channels_col, " \
        "height_col, width_col, %d, %d, %d, %d, %d, %d, %d, %d, %d);\n"%(
            n, input_id, 
            input_node.output_shape[2], input_node.output_shape[3],
            node.weight.shape[2], node.weight.shape[3],
            node.stride[0], node.stride[1], node.padding[0],
            node.padding[1], node.dilation[0], node.dilation[1],
            node.zero_x
        )
    code += "\n"

    # conv
    conv_index = get_conv2d_index(n, calculation_graph)
    calc_process = calc_process_with_parallel[conv_index]
    for pair_index, pair in enumerate(calc_process):
        left_line = pair[0]
        right_line = pair[1]
        left_line_type = None
        right_line_type = None
        if(not left_line is None):
            if(left_line[0][0:4] == "copy"):
                code += gen_copy(
                    n, 
                    calculation_graph,
                    im2col_shape,
                    divided_border,
                    sub_matrix_size,
                    calc_process_with_parallel,
                    pair_index,
                    0,       # select left or right
                )
            elif(left_line[0][0:7] == "set_dma"):
                code += gen_set_dma(
                    n, 
                    calculation_graph,
                    im2col_shape,
                    divided_border,
                    sub_matrix_size,
                    calc_process_with_parallel,
                    instr_analyse_result,
                    resource_analyse_result,
                    instruction_index,
                    pair_index,
                    0,       # select left or right
                )
            elif(left_line[0][0:4] == "load"):
                code += gen_load(
                    n, 
                    calculation_graph,
                    im2col_shape,
                    divided_border,
                    sub_matrix_size,
                    calc_process_with_parallel,
                    pair_index,
                    0,       # select left or right
                )
            elif(left_line[0][0:5] == "store"):
                code += gen_store(
                    n, 
                    calculation_graph,
                    im2col_shape,
                    divided_border,
                    sub_matrix_size,
                    calc_process_with_parallel,
                    pair_index,
                    0,       # select left or right
                )
        
        if(not right_line is None):
            if(right_line[0][0:4] == "copy"):
                code += gen_copy(
                    n, 
                    calculation_graph,
                    im2col_shape,
                    divided_border,
                    sub_matrix_size,
                    calc_process_with_parallel,
                    pair_index,
                    1,       # select left or right
                )
            elif(right_line[0][0:7] == "set_dma"):
                code += gen_set_dma(
                    n, 
                    calculation_graph,
                    im2col_shape,
                    divided_border,
                    sub_matrix_size,
                    calc_process_with_parallel,
                    instr_analyse_result,
                    resource_analyse_result,
                    instruction_index,
                    pair_index,
                    1,       # select left or right
                )
            elif(right_line[0][0:4] == "load"):
                code += gen_load(
                    n, 
                    calculation_graph,
                    im2col_shape,
                    divided_border,
                    sub_matrix_size,
                    calc_process_with_parallel,
                    pair_index,
                    1,       # select left or right
                )
            elif(right_line[0][0:5] == "store"):
                code += gen_store(
                    n, 
                    calculation_graph,
                    im2col_shape,
                    divided_border,
                    sub_matrix_size,
                    calc_process_with_parallel,
                    pair_index,
                    1,       # select left or right
                )


    code += "\n\n"

    return code



def gen_QMaxpool2d(n, calculation_graph):
    '''
    Generate maxpool2d
    '''
    code = ""

    # get node and some arguments
    node = calculation_graph[n]
    output_shape = node.output_shape

    # get input_node and some arguments
    input_id = node.input
    input_node = calculation_graph[input_id]
    input_shape = input_node.output_shape

    code += "// layer %d: %s\n" \
        "qmaxpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
        "%d, %d, %d, %d, %d, %d, %d, %d, %d);\n\n"%(
            n, node,
            n, node.input, input_shape[0], input_shape[1], input_shape[2],
            input_shape[3], node.kernel_size[0], node.kernel_size[1],
            node.stride[0], node.stride[1], node.padding[0], node.padding[1],
            node.dilation[0], node.dilation[1], node.zero
        )


    return code



def gen_QAvgpool2d(n, calculation_graph):
    '''
    Generate avgpool2d
    '''
    code = ""

    # get node and some arguments
    node = calculation_graph[n]
    output_shape = node.output_shape

    # get input_node and some arguments
    input_id = node.input
    input_node = calculation_graph[input_id]
    input_shape = input_node.output_shape

    code += "// layer %d: %s\n" \
        "qavgpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
        "%d, %d, %d, %d, %d, %d, %d);\n\n"%(
            n, node,
            n, node.input, input_shape[0], input_shape[1], input_shape[2], 
            input_shape[3], node.kernel_size[0], node.kernel_size[1],
            node.stride[0], node.stride[1], node.padding[0], node.padding[1],
            node.zero
        )


    return code



def gen_Relu(n, calculation_graph):
    '''
    generate relu code 
    #! Caution: part of ReLUs are fused into conv2d and dense
    '''
    code = ""
    
    # get node 
    node = calculation_graph[n]
    input_id = node.input

    # get input_node
    input_node = calculation_graph[input_id]
    if(type(input_node) == op.QConv2d or 
        type(input_node) == op.QDense):
        # ReLU fused into input_node
        code += "// layer %d: %s\n" \
            "// fused into input layer\n\n"%(
                n, node
            )
        return code
    
    # get data len
    length = 1
    for i in node.output_shape:
        length *= i
    
    code += "// layer %d: %s\n" \
        "qrelu(layer_%d_res, layer_%d_res, %d, %d);\n\n"%(
            n, node,
            n, node.input, length, node.zero
        )

    return code



def gen_QFlatten(n, calculation_graph):
    '''
    Generate Flatten code
    '''
    code = ""

    # get node
    node = calculation_graph[n]
    input_id = node.input

    length = 1
    for i in node.output_shape:
        length *= i
    
    code += "// layer %d: %s\n" \
        "qflatten(layer_%s_res, layer_%s_res, %d);\n\n"%(
            n, node,
            n, node.input, length
        )


    return code



def gen_QDense(
    n, 
    calculation_graph,
    instr_analyse_result,
):
    '''
    Generate fc code
    '''
    code = ""

    # get activation type
    activation = 0
    for node in calculation_graph:
        if(type(node) == op.QInput):
            continue
        elif(type(node) == op.QAdd or 
            type(node) == op.QConcat):
            if(node.input1 == n or node.input2 == n):
                next_node = node
                break
        else:
            if(node.input == n):
                next_node = node
                break
    if(type(next_node) == op.QRelu):
        activation = 1
    
    # get this node
    current_node = calculation_graph[n]

    # hidden and output channel
    hidden_channel = current_node.input_channel
    output_channel = current_node.output_channel

    # layer mux
    mux = 0
    for node_index, node in enumerate(calculation_graph):
        if(node_index >= n):
            break
        if(type(node) == op.QDense):
            mux += 1
    
    # list instr field(each item: (width, value))
    instr_field = [
        (instr_analyse_result["ps_calculation_type"], 1),
        (instr_analyse_result["ps_activation_for_fc"], activation),
        (instr_analyse_result["ps_hidden_channel_for_fc"], hidden_channel),
        (instr_analyse_result["ps_output_channel_for_fc"], output_channel),
        (instr_analyse_result["ps_layer_mux_for_fc"], mux),
        (instr_analyse_result["ps_bit_width_need"] - 
            instr_analyse_result["ps_bit_width_need_for_fc"], 0),
    ]
    
    # generate binary instruction code
    instruction_code = ""
    for item in instr_field:
        instruction_code += decimal_to_bin(item[1], item[0])
    
    # generate hexadecimal instruction code
    instruction_code = bin_to_hex(instruction_code)
    
    # divide into 32bit per word
    words = []
    for i in range(instr_analyse_result["ps_bit_width_need"] // 32):
        words.append(instruction_code[i*8 : i*8+8])
    
    # generate code
    code += "// layer %d: %s\n"%(n, current_node)
    code += "memcpy(layer_%d_in_dma, layer_%d_res, sizeof(uint8)*%d);\n"%(
        n, current_node.input, current_node.input_channel
    )
    for count, word in enumerate(words):
        code += "mmio[%d] = 0x%s;\n"%(count, word)
    code += "mmio[%d] = 0xFFFFFFFF;\n"%(count+1)
    code += """
PYNQ_writeDMA(&dma_r0, &layer_%d_in_shared, 0, sizeof(uint8)*%d);
PYNQ_waitForDMAComplete(&dma_r0, AXI_DMA_WRITE);
PYNQ_writeDMA(&dma_r0, &layer_%d_weight_shared, 0, sizeof(uint8)*%d);
PYNQ_waitForDMAComplete(&dma_r0, AXI_DMA_WRITE);
PYNQ_readDMA(&dma_w0, &layer_%d_out_shared, 0, sizeof(uint64)*%d);
PYNQ_waitForDMAComplete(&dma_w0, AXI_DMA_READ);\n"""%(
        n, current_node.input_channel,
        n, current_node.input_channel * current_node.output_channel,
        n, current_node.output_channel,
    )
    code += """
for(int i = 0; i<%d; i++) {
    layer_%d_res[i] = (uint8)(layer_%d_out_dma[i]&0x000000FF);
}
    \n"""%(
        current_node.output_channel,
        n, n, 
    )


    
    return code



def gen_QDropout(n, calculation_graph):
    '''
    Generate dropout code
    '''
    code = ""

    node = calculation_graph[n]

    length = 1
    for i in node.output_shape:
        length *= 1
    
    code += "// layer %d: %s\n" \
        "qdropout(layer_%d_res, layer_%d_res, %d);\n"%(
            n, node,
            n, node.input, length
        )


    return code



def gen_QOutput(
    n, 
    calculation_graph,
    output_count
):
    '''
    Generate output code
    '''
    code = ""

    node = calculation_graph[n]
    
    length = 1
    for i in node.output_shape:
        length *= 1
    
    code += "// layer %d: %s\n" \
        "qoutput(layer_%d_res, layer_%d_res, %d);\n"%(
            n, node,
            n, node.input, length
        )
    code += "result[%d] = layer_%d_res;\n"%(output_count, n)


    return code



def gen_QAdd(
    n, 
    calculation_graph,
    instr_analyse_result,
):
    '''
    Generate add code
    '''
    code = ""

    node = calculation_graph[n]
    
    # get data count
    length = 1
    for i in node.output_shape:
        length *= i
    
    # layer mux
    mux = 0
    for node_index, temp_node in enumerate(calculation_graph):
        if(node_index >= n):
            break
        if(type(temp_node) == op.QAdd):
            mux += 1
    
    # list instr field(each item: (width, value))
    instr_field = [
        (instr_analyse_result["ps_calculation_type"], 2),
        (instr_analyse_result["ps_total_count_for_add"], length),
        (instr_analyse_result["ps_layer_mux_for_add"], mux),
        (instr_analyse_result["ps_bit_width_need"] - 
            instr_analyse_result["ps_bit_width_need_for_add"], 0),
    ]

    # generate binary instruction code
    instruction_code = ""
    for item in instr_field:
        instruction_code += decimal_to_bin(item[1], item[0])
    
    # generate hexadecimal instruction code
    instruction_code = bin_to_hex(instruction_code)

    # divide into 32bit per word
    words = []
    for i in range(instr_analyse_result["ps_bit_width_need"] // 32):
        words.append(instruction_code[i*8 : i*8+8])
    
    # generate code
    code += "// layer %d: %s\n"%(n, node)
    code += "memcpy(layer_%d_in1_dma, layer_%d_res, sizeof(uint8)*%d);\n"%(
        n, node.input1, length
    )
    code += "memcpy(layer_%d_in2_dma, layer_%d_res, sizeof(uint8)*%d);\n"%(
        n, node.input2, length
    )
    for count, word in enumerate(words):
        code += "mmio[%d] = 0x%s;\n"%(count, word)
    code += "mmio[%d] = 0xFFFFFFFF;\n"%(count+1)
    code += """
PYNQ_writeDMA(&dma_r0, &layer_%d_in1_shared, 0, sizeof(uint8)*%d);
PYNQ_writeDMA(&dma_r1, &layer_%d_in2_shared, 0, sizeof(uint8)*%d);
PYNQ_readDMA(&dma_w0, &layer_%d_out_shared, 0, sizeof(uint8)*%d);
PYNQ_waitForDMAComplete(&dma_r0, AXI_DMA_WRITE);
PYNQ_waitForDMAComplete(&dma_r0, AXI_DMA_WRITE);
PYNQ_waitForDMAComplete(&dma_w0, AXI_DMA_READ);
memcpy(layer_%d_res, layer_%d_out_dma, sizeof(uint8)*%d);
    \n"""%(
        n, length,
        n, length,
        n, length,
        n, n, length
    )
    

    return code



def generate_calc(
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel,
    instr_analyse_result,
    resource_analyse_result,
    instruction_index,
):
    '''
    Generate calc
    '''
    code = ""
    code += "void calc(uint8 ** result, uint8 * data)\n"
    code += "{\n"
    code += "int kernel_h;\n"
    code += "int kernel_w;\n"
    code += "int dilation_h;\n"
    code += "int dilation_w;\n"
    code += "int height;\n"
    code += "int width;\n"
    code += "int pad_h;\n"
    code += "int pad_w;\n"
    code += "int stride_h;\n"
    code += "int stride_w;\n"
    code += "int channels;\n"
    code += "int dil_kernel_h;\n"
    code += "int dil_kernel_w;\n"
    code += "int height_col;\n"
    code += "int width_col;\n"
    code += "int channels_col;\n"
    code += "int yh;\n"
    code += "int yw;\n"
    code += "Fixed_point * fp_temp = Fixed_point_init(0);\n"
    code += "int * dst_ptr_int;\n"
    code += "int * src_ptr_int;\n"
    code += "uint8 * dst_ptr_uint8;\n"
    code += "uint8 * src_ptr_uint8;\n"

    # number of output_node
    output_count = 0

    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QInput):
            code += gen_QInput(n, calculation_graph)
        elif(type(node) == op.QConv2d):
            code += gen_QConv2d(
                n, 
                calculation_graph,
                im2col_shape,
                divided_border,
                sub_matrix_size,
                calc_process_with_parallel,
                instr_analyse_result,
                resource_analyse_result,
                instruction_index,
            )
        elif(type(node) == op.QMaxpool2d):
            code += gen_QMaxpool2d(n, calculation_graph)
        elif(type(node) == op.QAvgpool2d):
            code += gen_QAvgpool2d(n, calculation_graph)
        elif(type(node) == op.QRelu):
            code += gen_Relu(n, calculation_graph)
        elif(type(node) == op.QFlatten):
            code += gen_QFlatten(n, calculation_graph)
        elif(type(node) == op.QDense):
            code += gen_QDense(
                n, 
                calculation_graph,
                instr_analyse_result,
            )
        elif(type(node) == op.QDropout):
            code += gen_QDropout(n, calculation_graph)
        elif(type(node) == op.QOutput):
            code += gen_QOutput(n, calculation_graph, output_count)
            output_count += 1
        elif(type(node) == op.QAdd):
            code += gen_QAdd(
                n, 
                calculation_graph,
                instr_analyse_result,
            )
    code += "}\n"

    return code



def gen_call_lib_c(
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel,
    instr_analyse_result,
    resource_analyse_result,
    instruction_index,
):
    '''
    Generate call_lib.c to run on arm cpu
    '''
    code = ""

    # generate head file call
    code += "#include \"call_lib.h\""
    code += "\n\n"

    # generate nn calc functions
    code += gen_intnn()
    code += "\n\n"

    # declare pointer for weights
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
            type(node) == op.QDense):
            code += "uint8 * layer_%d_weight;\n"%(n)
            code += "int * layer_%d_bias;\n"%(n)
    code += "\n\n"

    # declare pointer for weight blocks
    code += declare_pointer_for_weight_blocks(
        calculation_graph,
        sub_matrix_size,
        calc_process_with_parallel
    )

    # declare pointer for medium results
    code += declare_pointer_for_medium_results(
        calculation_graph,
        im2col_shape,
        sub_matrix_size,
        calc_process_with_parallel
    )

    # declare bus
    code += declare_bus(
        BRAM_CTRL_0="0x40000000",
        AXI_DMA_R0="0x40400000",
        AXI_DMA_R1="0x40410000",
        AXI_DMA_W0="0x40420000",
    )

    # init
    code += generate_init(
        calculation_graph,
        im2col_shape,
        divided_border,
        sub_matrix_size,
        calc_process_with_parallel,
    )

    # calc
    code += generate_calc(
        calculation_graph,
        im2col_shape,
        divided_border,
        sub_matrix_size,
        calc_process_with_parallel,
        instr_analyse_result,
        resource_analyse_result,
        instruction_index,
    )
    

    return code



def gen_code(
    calculation_graph,
    im2col_shape,
    divided_border,
    submatrix_size,
    calc_process_with_parallel,
    instr_analyse_result,
    resource_analyse_result,
    instruction_index,
):
    '''
    Generate C code to run on arm cpu
    '''
    save_weight(calculation_graph)

    cmake_txt = gen_cmake()

    main_code = gen_main(
        calculation_graph=calculation_graph
    )

    call_lib_h_code = gen_call_lib_h()

    call_lib_c_code = gen_call_lib_c(
        calculation_graph,
        im2col_shape,
        divided_border,
        submatrix_size,
        calc_process_with_parallel,
        instr_analyse_result,
        resource_analyse_result,
        instruction_index,
    )

    fixed_point_h_code, fixed_point_c_code = gen_fixed_point_lib()

    return \
        cmake_txt, \
        main_code, \
        call_lib_h_code, \
        call_lib_c_code, \
        fixed_point_h_code, \
        fixed_point_c_code
