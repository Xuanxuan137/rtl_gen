

import op
import os
import numpy as np



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
    char sign;                      // 符号位 0为正，1为负
    unsigned int ivalue;            // 32位整数部分
    unsigned int fvalue;            // 16位小数部分。只使用低16bit
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
    unsigned int int_part = (int)f;     // 输入数据的整数部分数值
    float frac_part = f - int_part;     // 输入数据的小数部分数值

    this->ivalue = int_part;

    float frac_temp = 1.0;
    float frac_value = 0;               // 定点数中小数部分现已赋值的值
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
        this_fvalue = ~(this->fvalue | 0xffff0000) + 1;    // 先或0xffff0000使得fvalue的高16位(闲置的16位)变为1，这样取反后就全是0，然后加1
        int c = (this_fvalue >> 16) & 0x1;     // 记录取反加一后的进位
        this_fvalue = this_fvalue & 0x0000ffff; // 只保留低16位
        this_ivalue = ~(this->ivalue) + c;     // 整数部分取反之后不需要再加一，因为整数部分和小数部分是一体的，小数部分加一就行了
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
    if(this_ivalue & 0x80000000) {  // 如果最高位是1
        this_fvalue = this_fvalue - 1;
        int c = (this_fvalue >> 16) & 0x1;      // 向整数部分借位
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
    # 计算
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

    # copy weights
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
                row_exceed = cut_coordinate_row_upper - row_border
                col_exceed = cut_coordinate_col_upper - col_border
                # # original im2col block shape
                im2col_block_shape = (block_shape[0] - row_exceed,
                    block_shape[1] - col_exceed)
                
                # copy from `im2col_block` to `block`
                dst_start = copied_size
                src_start = cut_coordinate_row * col_border + cut_coordinate_col
                code += '''
weight_y_ptr = &layer_%d_weight_%d_dma[%d];
weight_x_ptr = &layer_%d_weight[%d];
                \n'''%(node_index, block_index, dst_start, node_index, src_start)

                copied_size += block_shape[0] * block_shape[1]
            block_index += 1
            # print("\n")
            # exit()


    # free original weights


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



def declare_bus():
    '''
    declare dma and bram
    '''
    code = ""
    code += "PYNQ_AXI_DMA dma_r0;\n"
    code += "PYNQ_AXI_DMA dma_r1;\n"
    code += "PYNQ_AXI_DMA dma_w0;\n"
    code += "volatile unsigned int * mmio;\n"
    code += "\n\n"

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


    code += "}\n"
    code += "\n\n"

    return code



def gen_call_lib_c(
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size,
    calc_process_with_parallel,
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
    code += declare_bus()

    # init
    code += generate_init(
        calculation_graph,
        im2col_shape,
        divided_border,
        sub_matrix_size,
        calc_process_with_parallel
    )
    

    return code



def gen_code(
    calculation_graph,
    im2col_shape,
    divided_border,
    submatrix_size,
    calc_process_with_parallel,
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
    )

    fixed_point_h_code, fixed_point_c_code = gen_fixed_point_lib()

    return \
        cmake_txt, \
        main_code, \
        call_lib_h_code, \
        call_lib_c_code, \
        fixed_point_h_code, \
        fixed_point_c_code