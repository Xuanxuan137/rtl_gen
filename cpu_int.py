
import enum
import op
import os

def get_conv2d_index(n, calculation_graph):
    '''
    计算calculation_graph中下标为n的节点是第几个conv2d
    从0开始计数
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


def gen_fixed_point_lib():
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

def gen_deploy(
    calculation_graph
):
    '''
    生成整体计算的代码
    1. 生成计算库, 及其头文件
    2. 生成主函数, 留空给用户填充
    3. 生成CMakeLists.txt
    '''
    
    # 0. 将权重保存到对应文件夹里(layer_%d_weight)
    if(not os.path.exists("output/int_deploy/weight")):
        os.system("mkdir -p output/int_deploy/weight")
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
           type(node) == op.QDense):
            weight_path = "output/int_deploy/weight/layer_%d_weight.bin"%(n)
            bias_path = "output/int_deploy/weight/layer_%d_bias.bin"%(n)
            node.weight.tofile(weight_path)
            node.bias.tofile(bias_path)
            
    # 1. 生成计算库
    # 1.1 生成fixed_point库
    fixed_point_h, fixed_point_c = gen_fixed_point_lib()
    with open("output/int_deploy/fixed_point.h", "w") as f:
        f.write(fixed_point_h)
    with open("output/int_deploy/fixed_point.c", "w") as f:
        f.write(fixed_point_c)

    # 1.2 生成calc_lib库
    # 生成头文件调用
    code = ""
    code += "#include \"call_lib.h\""
    code += "\n\n"

    # 生成nn计算函数
    code += gen_intnn()
    code += "\n\n"

    # 声明权重空间  (layer_%d_weight)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
           type(node) == op.QDense):
            code += "uint8 * layer_%d_weight;\n"%(n)
            code += "int * layer_%d_bias;\n"%(n)
    code += "\n\n"

    # 声明计算结果空间  (layer_%d_res, layer_%d_xcol)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "uint8 * layer_%d_xcol;\n"%(n)
        code += "uint8 * layer_%d_res;\n"%(n)
    code += "\n\n"

    # 声明Fixed_point (coe_%d, coe1_%d, coe2_%d)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
            type(node) == op.QDense):
            code += "Fixed_point * coe_%d;\n"%(n)
        elif(type(node) == op.QAdd or 
            type(node) == op.QConcat):
            code += "Fixed_point * coe1_%d;\n"%(n)
            code += "Fixed_point * coe2_%d;\n"%(n)

    # 生成init函数
    code += "void init()\n"
    code += "{\n"
    # 为权重分配空间
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
    # 读取权重数据
    code += "FILE * f;\n"
    code += "int l;\n"
    path = os.getcwd()
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "f = fopen(\"%s/output/int_deploy/weight/layer_%d_weight" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(uint8), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/int_deploy/weight/layer_%d_bias" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(int), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
        elif(type(node) == op.QDense):
            code += "f = fopen(\"%s/output/int_deploy/weight/layer_%d_weight" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(uint8), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/int_deploy/weight/layer_%d_bias" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(int), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
    code += "\n\n"
    # 为计算结果分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "layer_%d_xcol = (uint8*)malloc(sizeof(uint8)*%d);\n"%(
                n, node.weight.shape[1] * node.weight.shape[2] * 
                node.weight.shape[3] * node.output_shape[2] * 
                node.output_shape[3])
        length = 1
        for i in node.output_shape:
            length *= i
        code += "layer_%d_res = (uint8*)malloc(sizeof(uint8)*%d);\n"%(
            n, length)
    code += "\n\n"
    # 为Fixed_point分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or
            type(node) == op.QDense):
            code += "coe_%d = Fixed_point_init(%f);\n"%(n, node.coe)
        elif(type(node) == op.QAdd or
            type(node) == op.QConcat):
            code += "coe1_%d = Fixed_point_init(%f);\n"%(n, node.coe1)
            code += "coe2_%d = Fixed_point_init(%f);\n"%(n, node.coe2)
    code += "}\n"
    code += "\n\n"

    # 生成calc函数
    result_count = 0
    code += "void calc(uint8 ** result, uint8 * data)\n"
    code += "{\n"
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QAdd or
           type(node) == op.QConcat):
            input1_id = node.input1
            input2_id = node.input2
        elif(type(node) == op.QInput):
            pass
        else:
            input_id = node.input
        if(type(node) == op.QInput):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qinput(layer_%d_res, data, %d);\n"%(n, length)
        elif(type(node) == op.QConv2d):
            code += "qconv2d(layer_%d_res, layer_%d_res, layer_%d_xcol, " \
                "layer_%d_weight, layer_%d_bias, %d, %d, %d, %d, %d, %d, " \
                "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, coe_%d, " \
                "%d, %d, %d);\n"%(n, input_id, n, n, n, 
                calculation_graph[input_id].output_shape[0], 
                calculation_graph[input_id].output_shape[1],
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3],
                node.weight.shape[0], node.weight.shape[1],
                node.weight.shape[2], node.weight.shape[3],
                node.padding[0], node.padding[1], node.stride[0],
                node.stride[1], node.dilation[0], node.dilation[1],
                node.zero_x, node.zero_w, node.zero_b, node.zero_y,
                n, node.rshift, node.qmin, node.qmax)
        elif(type(node) == op.QMaxpool2d):
            code += "qmaxpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
                "%d, %d, %d, %d, %d, %d, %d, %d, %d);\n"%(n, input_id, 
                calculation_graph[input_id].output_shape[0],
                calculation_graph[input_id].output_shape[1],
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3],
                node.kernel_size[0], node.kernel_size[1],
                node.stride[0], node.stride[1], node.padding[0],
                node.padding[1], node.dilation[0], node.dilation[1],
                node.zero)
        elif(type(node) == op.QRelu):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qrelu(layer_%d_res, layer_%d_res, %d, %d);\n"%(
                n, input_id, length, node.zero)
        elif(type(node) == op.QFlatten):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qflatten(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
        elif(type(node) == op.QDense):
            code += "qfc(layer_%d_res, layer_%d_res, layer_%d_weight, " \
                "layer_%d_bias, %d, %d, %d, %d, %d, %d, coe_%d, %d, %d, " \
                "%d);\n"%(
                n, input_id, n, n, node.output_channel, node.input_channel,
                node.zero_x, node.zero_w, node.zero_b, node.zero_y,
                n, node.rshift, node.qmin, node.qmax)
        elif(type(node) == op.QDropout):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qdropout(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
        elif(type(node) == op.QOutput):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qoutput(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
            code += "result[%d] = layer_%d_res;\n"%(result_count, n)
            result_count += 1
        elif(type(node) == op.QAdd):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qadd(layer_%d_res, layer_%d_res, layer_%d_res, %d, " \
                "%d, %d, %d, coe1_%d, coe2_%d, %d, %d, %d, %d);\n"%(
                n, input1_id, input2_id, length, node.zero_x1, node.zero_x2, 
                node.zero_y, n, n, node.rshift1, node.rshift2, node.qmin, node.qmax)
        elif(type(node) == op.QConcat):
            # TODO: not supported yet
            raise TypeError("Not supported yet")
        elif(type(node) == op.QAvgpool2d):
            code += "qavgpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
            "%d, %d, %d, %d, %d, %d, %d);\n"%(n, input_id,
            calculation_graph[input_id].output_shape[0],
            calculation_graph[input_id].output_shape[1],
            calculation_graph[input_id].output_shape[2],
            calculation_graph[input_id].output_shape[3],
            node.kernel_size[0], node.kernel_size[1], node.stride[0],
            node.stride[1], node.padding[0], node.padding[1],
            node.zero)
        else:
            raise TypeError("Not supported yet")
    code += "}\n"




    with open("output/int_deploy/call_lib.c", "w") as f:
        f.write(code)
    
    # 2. 生成计算库头文件
    code = ""
    code += "#include <stdio.h>\n"
    code += "#include <stdlib.h>\n"
    code += "#include <string.h>\n"
    code += "#include <assert.h>\n"
    code += "#include <math.h>\n"
    code += "#include <cblas.h>\n"
    code += "#include \"fixed_point.h\"\n"

    code += "typedef unsigned char uint8;\n"
    code += "void init();\n"
    code += "void calc(uint8 ** result , uint8 * data);\n"
    code += "int argmax(uint8 * x, int len);\n"

    with open("output/int_deploy/call_lib.h", "w") as f:
        f.write(code)
    
    # 3. 生成主函数
    code = ""
    code += "#include \"call_lib.h\"\n"
    code += "int main(void)\n"
    code += "{\n"
    indent = "\t"
    # 加载数据集
    code += indent + "// load your dataset\n"
    code += "\n"
    # 初始化内存空间和权重数据
    code += indent + "// init weight and result memory\n"
    code += indent + "init();\n"
    code += "\n"
    # 声明结果空间
    code += indent + "// alloc space for result\n"
    code += indent + "uint8 ** result = (uint8**)malloc(sizeof(uint8*)*%d)" \
        ";\n"%(result_count)
    code += "\n"
    # 计算
    code += indent + "// calculation\n"
    code += indent + "calc(result, /* pointer to test data */);\n"
    code += "}\n"

    with open("output/int_deploy/main.c", "w") as f:
        f.write(code)
    
    # 4. 生成CMakelists.txt
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
    with open("output/int_deploy/CMakeLists.txt", "w") as f:
        f.write(code)


def gen_block_deploy(
    calculation_graph,
    im2col_shape,
    divided_border,
    sub_matrix_size
):
    '''
    生成用整数分块计算的代码
    1. 生成计算库, 及其头文件
    2. 生成主函数, 留空给用户填充
    3. 生成CMakeLists
    '''

    # 0. 将权重保存到对应文件夹里   (layer_%d_weight.bin)
    if(not os.path.exists("output/int_block_deploy/weight")):
        os.system("mkdir -p output/int_block_deploy/weight")
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
           type(node) == op.QDense):
            weight_path = "output/int_block_deploy/weight/layer_%d_weight" \
                ".bin"%(n)
            bias_path = "output/int_block_deploy/weight/layer_%d_bias.bin"%(n)
            node.weight.tofile(weight_path)
            node.bias.tofile(bias_path)
    
    # 1. 生成计算库
    # 1.1 生成fixed_point库
    fixed_point_h, fixed_point_c = gen_fixed_point_lib()
    with open("output/int_block_deploy/fixed_point.h", "w") as f:
        f.write(fixed_point_h)
    with open("output/int_block_deploy/fixed_point.c", "w") as f:
        f.write(fixed_point_c)
    
    # 1.2 生成calc_lib库
    # 生成头文件调用
    code = ""
    code += "#include \"call_lib.h\""
    code += "\n\n"

    # 生成nn计算函数
    code += gen_intnn()
    code += "\n\n"

    # 声明权重空间  (layer_%d_weight)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
           type(node) == op.QDense):
            code += "uint8 * layer_%d_weight;\n"%(n)
            code += "int * layer_%d_bias;\n"%(n)
    code += "\n\n"

    # 声明分块权重空间 (layer_%d_weight_%d_%d)
    for n, node in enumerate(calculation_graph):
        # 对于conv2d, 需要对权重切块
        if(type(node) != op.QConv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        for i in range(shape_A[0]):
            for j in range(shape_A[1]):
                code += "uint8 * layer_%d_weight_%d_%d;\n"%(n, i, j)
    code += "\n\n"

    # 声明计算结果空间  (layer_%d_res, layer_%d_xcol)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "uint8 * layer_%d_xcol;\n"%(n)
        code += "uint8 * layer_%d_res;\n"%(n)
    code += "\n\n"

    # 声明暂存计算结果空间 (layer_%d_res_temp)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "int * layer_%d_res_temp;\n"%(n)

    # 声明分块x_col空间
    for n, node in enumerate(calculation_graph):
        # 对于conv2d，需要对x_col切块
        if(type(node) != op.QConv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        for i in range(shape_B[0]):
            for j in range(shape_B[1]):
                code += "uint8 * layer_%d_xcol_%d_%d;\n"%(n, i, j)
    code += "\n\n"

    # 声明分块结果空间
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.QConv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        shape_C = (shape_A[0], shape_B[1])
        for i in range(shape_C[0]):
            for j in range(shape_C[1]):
                code += "int * layer_%d_res_%d_%d;\n"%(n, i, j)
    code += "\n\n"
    
    # 声明Fixed_point (coe_%d, coe1_%d, coe2_%d)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or 
            type(node) == op.QDense):
            code += "Fixed_point * coe_%d;\n"%(n)
        elif(type(node) == op.QAdd or 
            type(node) == op.QConcat):
            code += "Fixed_point * coe1_%d;\n"%(n)
            code += "Fixed_point * coe2_%d;\n"%(n)
    code += "\n\n"

    # 生成init函数
    code += "void init()\n"
    code += "{\n"
    # 为权重分配空间
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
    # 为分块权重分配空间
    for n, node in enumerate(calculation_graph):
        # 对于conv2d, 需要对权重切块
        if(type(node) != op.QConv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        for i in range(shape_A[0]):
            for j in range(shape_A[1]):
                code += "layer_%d_weight_%d_%d = (uint8*)malloc(sizeof" \
                    "(uint8)*%d);\n"%(n, i, j, 
                    sub_matrix_size_A[i][j][0] * sub_matrix_size_A[i][j][1])
    code += "\n\n"
    # 读取权重数据
    code += "FILE * f;\n"
    code += "int l;\n"
    path = os.getcwd()
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "f = fopen(\"%s/output/int_block_deploy/weight/" \
                "layer_%d_weight.bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(uint8), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/int_block_deploy/weight/" \
                "layer_%d_bias.bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(int), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
        elif(type(node) == op.QDense):
            code += "f = fopen(\"%s/output/int_block_deploy/weight/" \
                "layer_%d_weight.bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(uint8), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/int_block_deploy/weight/" \
                "layer_%d_bias.bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(int), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
    code += "\n\n"
    # 将权重数据复制到分块权重中
    code += "uint8 * dst_ptr;\n"
    code += "uint8 * src_ptr;\n"
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.QConv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        divided_border_A = divided_border[conv_index][0]
        divided_border_B = divided_border[conv_index][1]
        im2col_shape_A = im2col_shape[conv_index][0]
        im2col_shape_B = im2col_shape[conv_index][1]
        for i in range(shape_A[0]):
            for j in range(shape_A[1]):
                # 对当前块进行处理
                divided_border_index = i * shape_A[1] + j
                border = divided_border_A[divided_border_index]
                # 计算当前块的切分边界(这是可能超过矩阵实际边界的)
                start_h = border[0]
                end_h = border[1]
                start_w = border[2]
                end_w = border[3]
                # 读取当前层矩阵的实际边界
                actual_end_h = im2col_shape_A[0]
                actual_end_w = im2col_shape_A[1]
                # 计算需要复制的边界(超过边界的填0)
                copy_start_h = start_h
                copy_end_h = end_h if(end_h <= actual_end_h) else actual_end_h
                copy_start_w = start_w
                copy_end_w = end_w if(end_w <= actual_end_w) else actual_end_w
                # 生成代码
                #   # 计算copy起始点
                code += "dst_ptr = layer_%d_weight_%d_%d;\n"%(n, i, j)
                code += "src_ptr = &layer_%d_weight[%d];\n"%(
                    n, copy_start_h * actual_end_w + copy_start_w)
                #   # 遍历需要copy的每一行
                code += "for(int i = %d; i<%d; i++) {\n"%(
                    copy_start_h, copy_end_h)
                #   # copy左侧需要复制的部分，并移动dst_ptr
                code += "memcpy(dst_ptr, src_ptr, sizeof(uint8)*%d);\n"%(
                    copy_end_w - copy_start_w)
                code += "dst_ptr += %d;\n"%(copy_end_w - copy_start_w)
                #   # 如果原始矩阵不够，将右面部分填0，并移动dst_ptr
                if(actual_end_w < end_w):
                    code += "memset(dst_ptr, %d, sizeof(uint8)*%d);\n"%(
                        node.zero_w, end_w - copy_end_w)
                    code += "dst_ptr += %d;\n"%(end_w - copy_end_w)
                #   # 移动src_ptr(下移一行，即加上原始矩阵宽度)
                code += "src_ptr += %d;\n"%(actual_end_w)
                code += "}\n"
                #   # 如果end_h超过了actual_end_h，对剩下的行填0
                if(end_h > actual_end_h):
                    code += "memset(dst_ptr, %d, sizeof(uint8)*%d);\n"%(
                        node.zero_w, 
                        (end_h - actual_end_h) * (end_w - start_w))
                code += "\n"
    # 为计算结果分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            code += "layer_%d_xcol = (uint8*)malloc(sizeof(uint8)*%d);\n"%(
                n, node.weight.shape[1] * node.weight.shape[2] * 
                node.weight.shape[3] * node.output_shape[2] * 
                node.output_shape[3])
        length = 1
        for i in node.output_shape:
            length *= i
        code += "layer_%d_res = (uint8*)malloc(sizeof(uint8)*%d);\n"%(
            n, length)
    code += "\n\n"
    # 为暂存计算结果分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "layer_%d_res_temp = (int*)malloc(sizeof(int)*%d);\n"%(
                n, length)
    # 为分块x_col分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.QConv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        for i in range(shape_B[0]):
            for j in range(shape_B[1]):
                code += "layer_%d_xcol_%d_%d = (uint8*)malloc(sizeof(uint8)" \
                    "*%d);\n"%(n, i, j, 
                    sub_matrix_size_B[i][j][0] * sub_matrix_size_B[i][j][1])
    code += "\n\n"
    # 为分块结果分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.QConv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        shape_C = (shape_A[0], shape_B[1])
        for i in range(shape_C[0]):
            for j in range(shape_C[1]):
                code += "layer_%d_res_%d_%d = (int*)malloc(sizeof(int)" \
                    "*%d);\n"%(n, i, j, 
                    sub_matrix_size_A[i][0][0] * sub_matrix_size_B[0][j][1])
    # 为Fixed_point分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QConv2d or
            type(node) == op.QDense):
            code += "coe_%d = Fixed_point_init(%f);\n"%(n, node.coe)
        elif(type(node) == op.QAdd or
            type(node) == op.QConcat):
            code += "coe1_%d = Fixed_point_init(%f);\n"%(n, node.coe1)
            code += "coe2_%d = Fixed_point_init(%f);\n"%(n, node.coe2)
    code += "\n\n"
    code += "}\n"

    # 生成calc函数
    result_count = 0
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
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.QAdd or
           type(node) == op.QConcat):
            input1_id = node.input1
            input2_id = node.input2
        elif(type(node) == op.QInput):
            pass
        else:
            input_id = node.input
        if(type(node) == op.QInput):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qinput(layer_%d_res, data, %d);\n"%(n, length)
        elif(type(node) == op.QConv2d):
            code += "\n"
            # im2col
            code += "channels_col = %d * %d * %d;\n"%(
                calculation_graph[input_id].output_shape[1],
                node.weight.shape[2], node.weight.shape[3])
            code += "dil_kernel_h = (%d - 1) * %d + 1;\n"%(
                node.weight.shape[2], node.dilation[0])
            code += "dil_kernel_w = (%d - 1) * %d + 1;\n"%(
                node.weight.shape[3], node.dilation[1])
            code += "height_col = (%d + 2 * %d - dil_kernel_h) / %d + 1;\n"%(
                calculation_graph[input_id].output_shape[2], 
                node.padding[0], node.stride[0])
            code += "width_col = (%d + 2 * %d - dil_kernel_w) / %d + 1;\n"%(
                calculation_graph[input_id].output_shape[3],
                node.padding[1], node.stride[1])
            code += "qim2col(layer_%d_xcol, layer_%d_res, %d, %d, channels_col, " \
                "height_col, width_col, %d, %d, %d, %d, %d, %d, %d, %d, %d);\n"%(
                n, input_id, 
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3],
                node.weight.shape[2], node.weight.shape[3],
                node.stride[0], node.stride[1], node.padding[0],
                node.padding[1], node.dilation[0], node.dilation[1],
                node.zero_x)
            # im2col结果复制到分块x_col
            conv_index = get_conv2d_index(n, calculation_graph)
            sub_matrix_size_A = sub_matrix_size[conv_index][0]
            sub_matrix_size_B = sub_matrix_size[conv_index][1]
            shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
            shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
            divided_border_A = divided_border[conv_index][0]
            divided_border_B = divided_border[conv_index][1]
            im2col_shape_A = im2col_shape[conv_index][0]
            im2col_shape_B = im2col_shape[conv_index][1]
            for i in range(shape_B[0]):
                for j in range(shape_B[1]):
                    # 对当前块进行处理
                    divided_border_index = i * shape_B[1] + j
                    border = divided_border_B[divided_border_index]
                    # 计算当前块的切分边界(这是可能超过矩阵实际边界的)
                    start_h = border[0]
                    end_h = border[1]
                    start_w = border[2]
                    end_w = border[3]
                    # 读取当前层矩阵的实际边界
                    actual_end_h = im2col_shape_B[0]
                    actual_end_w = im2col_shape_B[1]
                    # 计算需要复制的边界(超过边界的填0)
                    copy_start_h = start_h
                    copy_end_h = end_h if(end_h <= actual_end_h) else actual_end_h
                    copy_start_w = start_w
                    copy_end_w = end_w if(end_w <= actual_end_w) else actual_end_w
                    # 生成代码
                    #   # 计算copy起始点
                    code += "dst_ptr_uint8 = layer_%d_xcol_%d_%d;\n"%(n, i, j)
                    code += "src_ptr_uint8 = &layer_%d_xcol[%d];\n"%(
                        n, copy_start_h * actual_end_w + copy_start_w)
                    #   # 遍历需要copy的每一行
                    code += "for(int i = %d; i<%d; i++) {\n"%(
                        copy_start_h, copy_end_h)
                    #   # copy左侧需要复制的部分，并移动dst_ptr
                    code += "memcpy(dst_ptr_uint8, src_ptr_uint8, " \
                        "sizeof(uint8)*%d);\n"%(
                        copy_end_w - copy_start_w)
                    code += "dst_ptr_uint8 += %d;\n"%(
                        copy_end_w - copy_start_w)
                    #   # 如果原始矩阵不够，将右面部分填0，并移动dst_ptr
                    if(actual_end_w < end_w):
                        code += "memset(dst_ptr_uint8, %d, sizeof(uint8)*" \
                            "%d);\n"%(
                            node.zero_x, end_w - copy_end_w)
                        code += "dst_ptr_uint8 += %d;\n"%(end_w - copy_end_w)
                    #   # 移动src_ptr(下移一行，即加上原始矩阵宽度)
                    code += "src_ptr_uint8 += %d;\n"%(actual_end_w)
                    code += "}\n"
                    #   # 如果end_h超过了actual_end_h，对剩下的行填0
                    if(end_h > actual_end_h):
                        code += "memset(dst_ptr_uint8, %d, sizeof(uint8)*" \
                            "%d);\n"%(
                            node.zero_x,
                            (end_h - actual_end_h) * (end_w - start_w))
                    code += "\n"
            # 分块weight和分块x_col计算，结果存入分块res
            shape_C = (shape_A[0], shape_B[1])
            for i in range(shape_C[0]):
                for j in range(shape_C[1]):
                    code += "memset(layer_%d_res_%d_%d, 0, sizeof(int)*%d)" \
                        ";\n"%(n, i, j, 
                    sub_matrix_size_A[i][0][0] * sub_matrix_size_B[0][j][1])
            M = shape_C[0]
            N = shape_C[1]
            K = shape_A[1]
            for i in range(M):
                for j in range(N):
                    for k in range(K):
                        # code += "cblas_sgemm(CblasRowMajor, CblasNoTrans, "\
                        #     "CblasNoTrans, %d, %d, %d, 1.0f, " \
                        #     "layer_%d_weight_%d_%d, %d, " \
                        #     "layer_%d_xcol_%d_%d, %d, 1.0f, " \
                        #     "layer_%d_res_%d_%d, %d);\n"%(
                        #     sub_matrix_size_A[i][0][0],
                        #     sub_matrix_size_B[0][j][1],
                        #     sub_matrix_size_A[0][k][1],
                        #     n, i, k, sub_matrix_size_A[0][k][1],
                        #     n, k, j, sub_matrix_size_B[0][j][1],
                        #     n, i, j, sub_matrix_size_B[0][j][1]
                        #     )
                        code += "igemm(layer_%d_res_%d_%d, " \
                            "layer_%d_weight_%d_%d, layer_%d_xcol_%d_%d, " \
                            "%d, %d, %d, 1, %d, %d);\n"%(
                                n, i, j, n, i, k, n, k, j,
                                sub_matrix_size_A[i][0][0],
                                sub_matrix_size_B[0][j][1],
                                sub_matrix_size_A[0][k][1],
                                node.zero_w, node.zero_x
                            )
            # 分块res复制到res_temp
            for i in range(shape_C[0]):
                for j in range(shape_C[1]):
                    # 对当前块进行处理
                    # C的第i行对应A的第i行，列不影响
                    divided_border_index_A = i * shape_A[1]
                    # C的第j列队迎B的第j列，行不影响
                    divided_border_index_B = j
                    # 计算当前块的border
                    border_A = divided_border_A[divided_border_index_A]
                    border_B = divided_border_B[divided_border_index_B]
                    border = [border_A[0], border_A[1], 
                        border_B[2], border_B[3]]
                    # 计算当前块的切分边界(这是可能超过矩阵实际边界的)
                    start_h = border[0]
                    end_h = border[1]
                    start_w = border[2]
                    end_w = border[3]
                    # 读取当前层矩阵的实际边界
                    actual_end_h = im2col_shape_A[0]
                    actual_end_w = im2col_shape_B[1]
                    # 计算需要复制的边界(超过边界的填0)
                    copy_start_h = start_h
                    copy_end_h = end_h if(end_h <= actual_end_h) \
                        else actual_end_h
                    copy_start_w = start_w
                    copy_end_w = end_w if(end_w <= actual_end_w) \
                        else actual_end_w
                    # 生成代码
                    #   # 计算copy起始点
                    code += "dst_ptr_int = &layer_%d_res_temp[%d];\n"%(
                        n, copy_start_h * actual_end_w + copy_start_w)
                    code += "src_ptr_int = layer_%d_res_%d_%d;\n"%(n, i, j)
                    #   # 遍历需要copy的每一行
                    code += "for(int i = %d; i<%d; i++) {\n"%(
                        copy_start_h, copy_end_h)
                    #   # copy左侧需要复制的部分，
                    #   #   并移动dst_ptr(下移一行,即加上原始矩阵宽度)
                    code += "memcpy(dst_ptr_int, src_ptr_int, sizeof(int)*%d);\n"%(
                        copy_end_w - copy_start_w)
                    code += "dst_ptr_int += %d;\n"%(actual_end_w)
                    #   # 移动src_ptr(下移一行，即加上块矩阵宽度)
                    code += "src_ptr_int += %d;\n"%(end_w - start_w)
                    code += "}\n"
                    #   # 对于块矩阵超过原始矩阵的部分，舍去不要
            # 加上bias，后处理并存入res
            code += "for(int i = 0; i<%d; i++) {\n"%(im2col_shape_A[0])
            code += "for(int j = 0; j<%d; j++) {\n"%(im2col_shape_B[1])
            code += "int temp = layer_%d_res_temp[i*%d+j];\n"%(
                n, im2col_shape_B[1])
            code += "temp += layer_%d_bias[i] - %d;\n"%(n, node.zero_b)
            code += "fp_temp->assign(fp_temp, temp);\n"
            code += "fp_temp->mult(fp_temp, coe_%d);\n"%(n)
            code += "int t = fp_temp->to_int(fp_temp);\n"
            code += "temp = (t >> %d) + %d;\n"%(node.rshift, node.zero_y)
            code += "layer_%d_res[i*%d+j] = (uint8)(\n"%(n, im2col_shape_B[1])
            code += "(temp < 0) ? 0 : \n"
            code += "(temp > 255) ? 255 : \n"
            code += "temp\n"
            code += ");\n"
            # code += "layer_%d_res[i*%d+j] += layer_%d_bias[i];\n"%(
            #     n, im2col_shape_B[1], n)
            code += "}\n"
            code += "}\n"
            code += "\n"
        elif(type(node) == op.QMaxpool2d):
            code += "qmaxpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
                "%d, %d, %d, %d, %d, %d, %d, %d, %d);\n"%(n, input_id, 
                calculation_graph[input_id].output_shape[0],
                calculation_graph[input_id].output_shape[1],
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3],
                node.kernel_size[0], node.kernel_size[1],
                node.stride[0], node.stride[1], node.padding[0],
                node.padding[1], node.dilation[0], node.dilation[1],
                node.zero)
        elif(type(node) == op.QRelu):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qrelu(layer_%d_res, layer_%d_res, %d, %d);\n"%(
                n, input_id, length, node.zero)
        elif(type(node) == op.QFlatten):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qflatten(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
        elif(type(node) == op.QDense):
            code += "qfc(layer_%d_res, layer_%d_res, layer_%d_weight, " \
                "layer_%d_bias, %d, %d, %d, %d, %d, %d, coe_%d, %d, %d, %d)" \
                ";\n"%(
                n, input_id, n, n, node.output_channel, node.input_channel,
                node.zero_x, node.zero_w, node.zero_b, node.zero_y, 
                n, node.rshift, node.qmin, node.qmax)
        elif(type(node) == op.QDropout):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qdropout(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
        elif(type(node) == op.QOutput):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qoutput(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
            code += "result[%d] = layer_%d_res;\n"%(result_count, n)
            result_count += 1
        elif(type(node) == op.QAdd):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "qadd(layer_%d_res, layer_%d_res, layer_%d_res, %d, %d, " \
                "%d, %d, coe1_%d, coe2_%d, %d, %d, %d, %d);\n"%(
                n, input1_id, input2_id, length, node.zero_x1, node.zero_x2,
                node.zero_y, n, n, node.rshift1, node.rshift2, node.qmin, node.qmax)
        elif(type(node) == op.QConcat):
            # TODO: not supported yet
            raise TypeError("Not supported yet")
        elif(type(node) == op.QAvgpool2d):
            code += "qavgpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
            "%d, %d, %d, %d, %d, %d, %d);\n"%(n, input_id,
            calculation_graph[input_id].output_shape[0],
            calculation_graph[input_id].output_shape[1],
            calculation_graph[input_id].output_shape[2],
            calculation_graph[input_id].output_shape[3],
            node.kernel_size[0], node.kernel_size[1], node.stride[0],
            node.stride[1], node.padding[0], node.padding[1], node.zero)
        else:
            raise TypeError("Not supported yet")
    
    code += "}\n"








    with open("output/int_block_deploy/call_lib.c", "w") as f:
        f.write(code)
    
    # 2. 生成计算库头文件
    code = ""
    code += "#include <stdio.h>\n"
    code += "#include <stdlib.h>\n"
    code += "#include <string.h>\n"
    code += "#include <assert.h>\n"
    code += "#include <math.h>\n"
    code += "#include <cblas.h>\n"
    code += "#include \"fixed_point.h\"\n"

    code += "typedef unsigned char uint8;\n"
    code += "void init();\n"
    code += "void calc(uint8 ** result , uint8 * data);\n"
    code += "int argmax(uint8 * x, int len);\n"

    with open("output/int_block_deploy/call_lib.h", "w") as f:
        f.write(code)
    
    # 3. 生成主函数
    code = ""
    code += "#include \"call_lib.h\"\n"
    code += "int main(void)\n"
    code += "{\n"
    indent = "\t"
    # 加载数据集
    code += indent + "// load your dataset\n"
    code += "\n"
    # 初始化内存空间和权重数据
    code += indent + "// init weight and result memory\n"
    code += indent + "init();\n"
    code += "\n"
    # 声明结果空间
    code += indent + "// alloc space for result\n"
    code += indent + "uint8 ** result = (uint8**)malloc(sizeof(uint8*)*%d)" \
        ";\n"%(result_count)
    code += "\n"
    # 计算
    code += indent + "// calculation\n"
    code += indent + "calc(result, /* pointer to test data */);\n"
    code += "}\n"

    with open("output/int_block_deploy/main.c", "w") as f:
        f.write(code)
    
    # 4. 生成CMakelists.txt
    code = ""
    code += """
cmake_minimum_required(VERSION 3.5)
project(int_block_deploy)
set(CMAKE_C_FLAGS "-O3 -Wall -W -pthread -fopenmp")
set(CMAKE_BUILD_TYPE "Release")
include_directories(
    /opt/OpenBLAS/include
)
link_directories(
    /opt/OpenBLAS/lib
)
add_executable(int_block_deploy main.c fixed_point.c call_lib.c)
target_link_libraries(int_block_deploy openblas pthread m)    
"""
    with open("output/int_block_deploy/CMakeLists.txt", "w") as f:
        f.write(code)



def gen_code(
    im2col_shape,
    calculation_graph,
    divided_border,
    submatrix_size,
    calc_process,
    model_dir
):
    '''
    生成cpu整数计算代码, 验证正确性
    '''

    # 生成整体计算的代码
    gen_deploy(
        calculation_graph
    )

    # 生成分块计算的代码
    gen_block_deploy(
        calculation_graph,
        im2col_shape,
        divided_border,
        submatrix_size
    )