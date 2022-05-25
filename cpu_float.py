

import os

import graph
import op
import numpy as np


def get_conv2d_index(n, calculation_graph):
    '''
    计算calculation_graph中下标为n的节点是第几个conv2d
    从0开始计数
    '''
    if(type(calculation_graph[n]) != op.Conv2d):
        raise TypeError("Calculation_graph[n] is not Conv2d")
    i = 0
    count = -1
    while(i <= n):
        if(type(calculation_graph[i]) == op.Conv2d):
            count += 1
        i += 1
    return count


def gen_blasnn():
    code = \
"""
void input(float * y, float * x, int len)
{
    memcpy(y, x, sizeof(float)*len);
}

void padding(float * y, float * x, int xn, int xc, int xh, int xw, int ph, int pw)
{
    int padded_w = xw + 2*pw;
    float * y_ptr = y;
    float * x_ptr = x;
    for(int n = 0; n<xn; n++) {
        for(int c = 0; c<xc; c++) {
            for(int h = 0; h<ph; h++) {
                memset(y_ptr, 0, sizeof(float)*padded_w);
                y_ptr += padded_w;
            }
            for(int h = 0; h<xh; h++) {
                memset(y_ptr, 0, sizeof(float)*pw);
                y_ptr += pw;
                memcpy(y_ptr, x_ptr, sizeof(float)*xw);
                y_ptr += xw;
                x_ptr += xw;
                memset(y_ptr, 0, sizeof(float)*pw);
                y_ptr += pw;
            }
            for(int h = 0; h<ph; h++) {
                memset(y_ptr, 0, sizeof(float)*padded_w);
                y_ptr += padded_w;
            }
        }
    }
}

void im2col(float * data_col, float * data_im, int height, int width, int channels_col, 
            int height_col, int width_col, int kernel_h, int kernel_w, int stride_h, int stride_w, 
            int pad_h, int pad_w, int dilation_h, int dilation_w)
{
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
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
                    data_col[row_offset + w] = 0.;
                }
            }
        }
    }
}

void conv2d(float * y, float * x, float * x_col, float * weight, float * bias, int xn, int xc, int xh, int xw, 
            int ko, int ki, int kh, int kw, int ph, int pw, int sh, int sw, int dh, int dw)
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

    im2col(x_col, x, xh, xw, channels_col, height_col, width_col, kh, kw, sh, sw, ph, pw, dh, dw);
   
    cblas_sgemm(CblasRowMajor, CblasNoTrans,CblasNoTrans,
                ko, yh*yw, ki*kh*kw, 1.0f,
                weight,ki*kh*kw, x_col,yh*yw, 0.0f, y,yh*yw);

    for(int i = 0; i<ko; i++) {
        for(int j = 0; j<yh*yw; j++) {
            y[i*yh*yw + j] += bias[i];
        }
    }
}

void relu(float * y, float * x, int len)
{
    for(int i = 0; i<len; i++) {
        y[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void maxpool2d(float * y, float * x, int xn, int xc, int xh, int xw, int kh, int kw, int sh, int sw, 
               int ph, int pw, int dh, int dw)
{
    int padded_h = xh + ph*2;
    int padded_w = xw + pw*2;
    float * padded = (float*)malloc(sizeof(float)*xn*xc*padded_h*padded_w);
    padding(padded, x, xn, xc, xh, xw, ph, pw);
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
                    float max = padded[
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

void avgpool2d(float * y, float * x, int xn, int xc, int xh, int xw, int kh, int kw, int sh, int sw, 
               int ph, int pw)
{
    int padded_h = xh + ph*2;
    int padded_w = xw + pw*2;
    float * padded = (float*)malloc(sizeof(float)*xn*xc*padded_h*padded_w);
    padding(padded, x, xn, xc, xh, xw, ph, pw);
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
                    float sum = 0;
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
                    ] = sum / (float)(kh * kw);
                }
            }
        }
    }
    free(padded);
}

void flatten(float * y, float * x, int len)
{
    memcpy(y, x, sizeof(float)*len);
}

void fc(float * y, float * x, float * weight, float * bias, int oc, int ic)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans,CblasNoTrans,
                1, oc, ic, 1.0f,
                x,ic, weight,oc, 0.0f, y,oc);
    
    for(int o = 0; o<oc; o++) {
        y[o] += bias[o];
    }
}

void dropout(float * y, float * x, float p, int len)
{
    for(int i = 0; i<len; i++) {
        y[i] = x[i] * p;
    }
}

void output(float * y, float * x, int len)
{
    memcpy(y, x, sizeof(float)*len);
}

void add(float * y, float * x1, float * x2, int len)
{
    for(int i = 0; i<len; i++) {
        y[i] = x1[i] + x2[i];
    }
}

int argmax(float * x, int len)
{
    float temp = x[0];
    int index = 0;
    for(int i = 0; i<len; i++) {
        if(x[i] > temp) {
            temp = x[i];
            index = i;
        }
    }
    return index;
}

void batch_norm2d(float * y, float * x, float * running_mean, float * running_var, 
                 float * weight, float * bias, float eps, int xn, int xc, int xh, int xw)
{
    float * E_x = (float*)malloc(sizeof(float)*xc);
    float * Var_x = (float*)malloc(sizeof(float)*xc);
    for(int i = 0; i<xc; i++) {
        E_x[i] = running_mean[i];
        Var_x[i] = running_var[i];
    }
    for(int n = 0; n<xn; n++) {
        for(int c = 0; c<xc; c++) {
            for(int h = 0; h<xh; h++) {
                for(int w = 0; w<xw; w++) {
                    x[
                        n * xc * xh * xw + 
                        c * xh * xw + 
                        h * xw + 
                        w
                    ] -= E_x[c];
                }
            }
        }
    }
    for(int i = 0; i<xc; i++) {
        Var_x[i] += eps;
        Var_x[i] = sqrt(Var_x[i]);
    }
    for(int n = 0; n<xn; n++) {
        for(int c = 0; c<xc; c++) {
            for(int h = 0; h<xh; h++) {
                for(int w = 0; w<xw; w++) {
                    y[
                        n * xc * xh * xw + 
                        c * xh * xw + 
                        h * xw + 
                        w 
                    ] = (x[
                        n * xc * xh * xw + 
                        c * xh * xw + 
                        h * xw + 
                        w 
                    ] / Var_x[c]) * weight[c] + bias[c];
                }
            }
        }
    }
    free(E_x);
    free(Var_x);
}
"""
    return code


def gen_blas_deploy(
    float_model_dir
):
    '''
    生成用OpenBlas计算的原始代码
    1. 生成计算库, 及其头文件
    2. 生成主函数, 留空给用户填充
    3. 生成CMakeLists
    '''
    calculation_graph = graph.read_calculation_graph(float_model_dir)
    graph.infer_output_shape(calculation_graph)

    # 0. 将权重保存到对应文件夹里  (layer_%d_weight.bin)
    if(not os.path.exists("output/blas_deploy/weight")):
        os.system("mkdir -p output/blas_deploy/weight")
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or 
           type(node) == op.Dense):
            weight_path = "output/blas_deploy/weight/layer_%d_weight.bin"%(n)
            bias_path = "output/blas_deploy/weight/layer_%d_bias.bin"%(n)
            node.weight.tofile(weight_path)
            node.bias.tofile(bias_path)
        elif(type(node) == op.Batch_norm2d):
            weight_path = "output/blas_deploy/weight/layer_%d_weight.bin"%(n)
            bias_path = "output/blas_deploy/weight/layer_%d_bias.bin"%(n)
            running_mean_path = "output/blas_deploy/weight/" \
                "layer_%d_running_mean.bin"%(n)
            running_var_path = "output/blas_deploy/weight/" \
                "layer_%d_running_var.bin"%(n)
            node.weight.tofile(weight_path)
            node.bias.tofile(bias_path)
            node.running_mean.tofile(running_mean_path)
            node.running_var.tofile(running_var_path)
    
    # 1. 生成计算库
    # 生成头文件调用
    code = ""
    code += "#include \"call_lib.h\""
    code += "\n\n"

    # 生成nn计算函数
    code += gen_blasnn()
    code += "\n\n"

    # 声明权重空间  (layer_%d_weight)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or 
           type(node) == op.Dense):
            code += "float * layer_%d_weight;\n"%(n)
            code += "float * layer_%d_bias;\n"%(n)
        elif(type(node) == op.Batch_norm2d):
            code += "float * layer_%d_weight;\n"%(n)
            code += "float * layer_%d_bias;\n"%(n)
            code += "float * layer_%d_running_mean;\n"%(n)
            code += "float * layer_%d_running_var;\n"%(n)
    code += "\n\n"
    
    # 声明计算结果空间  (layer_%d_res, layer_%d_xcol)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d):
            code += "float * layer_%d_xcol;\n"%(n)
        code += "float * layer_%d_res;\n"%(n)
    code += "\n\n"
    
    # 生成init函数
    code += "void init()\n"
    code += "{\n"
    # 为权重分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d):
            code += "layer_%d_weight = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "layer_%d_bias = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.bias.shape[0])
        elif(type(node) == op.Dense):
            code += "layer_%d_weight = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1])
            code += "layer_%d_bias = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.bias.shape[0])
        elif(type(node) == op.Batch_norm2d):
            code += "layer_%d_weight = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.weight.shape[0])
            code += "layer_%d_bias = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.bias.shape[0])
            code += "layer_%d_running_mean = (float*)malloc(sizeof(float)*%d" \
                ");\n"%(n, node.running_mean.shape[0])
            code += "layer_%d_running_var = (float*)malloc(sizeof(float)*%d" \
                ");\n"%(n, node.running_var.shape[0])
    code += "\n\n"
    # 读取权重数据
    code += "FILE * f;\n"
    code += "int l;\n"
    path = os.getcwd()
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d):
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_weight" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(float), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_bias" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(float), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
        elif(type(node) == op.Dense):
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_weight" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(float), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_bias" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(float), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
        elif(type(node) == op.Batch_norm2d):
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_weight" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(float), %d, f);\n"%(
                n, node.weight.shape[0])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_bias" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(float), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/" \
                "layer_%d_running_mean.bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_running_mean, sizeof(float), %d," \
                " f);\n"%(n, node.running_mean.shape[0])
            code += "assert(l == %d);\n"%(
                node.running_mean.shape[0])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/" \
                "layer_%d_running_var.bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_running_var, sizeof(float), %d," \
                " f);\n"%(n, node.running_var.shape[0])
            code += "assert(l == %d);\n"%(
                node.running_var.shape[0])
            code += "fclose(f);\n"
    code += "\n\n"
    # 为计算结果分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d):
            code += "layer_%d_xcol = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.weight.shape[1] * node.weight.shape[2] * 
                node.weight.shape[3] * node.output_shape[2] * 
                node.output_shape[3])
        length = 1
        for i in node.output_shape:
            length *= i
        code += "layer_%d_res = (float*)malloc(sizeof(float)*%d);\n"%(
            n, length)
    code += "}\n"
    code += "\n\n"

    # 生成calc函数
    result_count = 0
    code += "void calc(float ** result, float * data)\n"
    code += "{\n"
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Add or
           type(node) == op.Concat):
            input1_id = node.input1
            input2_id = node.input2
        elif(type(node) == op.Input):
            pass
        else:
            input_id = node.input
        if(type(node) == op.Input):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "input(layer_%d_res, data, %d);\n"%(n, length)
        elif(type(node) == op.Conv2d):
            code += "conv2d(layer_%d_res, layer_%d_res, layer_%d_xcol, " \
                "layer_%d_weight, layer_%d_bias, %d, %d, %d, %d, %d, %d, " \
                "%d, %d, %d, %d, %d, %d, %d, %d);\n"%(n, input_id, n, n, n, 
                calculation_graph[input_id].output_shape[0], 
                calculation_graph[input_id].output_shape[1],
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3],
                node.weight.shape[0], node.weight.shape[1],
                node.weight.shape[2], node.weight.shape[3],
                node.padding[0], node.padding[1], node.stride[0],
                node.stride[1], node.dilation[0], node.dilation[1])
        elif(type(node) == op.Maxpool2d):
            code += "maxpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
                "%d, %d, %d, %d, %d, %d, %d, %d);\n"%(n, input_id, 
                calculation_graph[input_id].output_shape[0],
                calculation_graph[input_id].output_shape[1],
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3],
                node.kernel_size[0], node.kernel_size[1],
                node.stride[0], node.stride[1], node.padding[0],
                node.padding[1], node.dilation[0], node.dilation[1])
        elif(type(node) == op.Relu):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "relu(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
        elif(type(node) == op.Flatten):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "flatten(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
        elif(type(node) == op.Dense):
            code += "fc(layer_%d_res, layer_%d_res, layer_%d_weight, " \
                "layer_%d_bias, %d, %d);\n"%(
                n, input_id, n, n, node.output_channel, node.input_channel)
        elif(type(node) == op.Dropout):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "dropout(layer_%d_res, layer_%d_res, %f, %d);\n"%(
                n, input_id, node.p, length)
        elif(type(node) == op.Output):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "output(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
            code += "result[%d] = layer_%d_res;\n"%(result_count, n)
            result_count += 1
        elif(type(node) == op.Add):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "add(layer_%d_res, layer_%d_res, layer_%d_res, %d);\n"%(
                n, input1_id, input2_id, length)
        elif(type(node) == op.Concat):
            # TODO: not supported yet
            raise TypeError("Not supported yet")
        elif(type(node) == op.Avgpool2d):
            code += "avgpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
            "%d, %d, %d, %d, %d, %d);\n"%(n, input_id,
            calculation_graph[input_id].output_shape[0],
            calculation_graph[input_id].output_shape[1],
            calculation_graph[input_id].output_shape[2],
            calculation_graph[input_id].output_shape[3],
            node.kernel_size[0], node.kernel_size[1], node.stride[0],
            node.stride[1], node.padding[0], node.padding[1])
        elif(type(node) == op.Batch_norm2d):
            code += "batch_norm2d(layer_%d_res, layer_%d_res, " \
                "layer_%d_running_mean, layer_%d_running_var, " \
                "layer_%d_weight, layer_%d_bias, %f, %d, %d, %d, %d);\n"%(
                n, input_id, n, n, n, n, node.eps, 
                calculation_graph[input_id].output_shape[0],
                calculation_graph[input_id].output_shape[1],
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3])
        else:
            raise TypeError("Not supported yet")
    code += "}\n"
    
    if(not os.path.exists("output/blas_deploy")):
        os.system("mkdir output/blas_deploy")
    with open("output/blas_deploy/call_lib.c", "w") as f:
        f.write(code)
    
    # 2. 生成计算库头文件
    code = ""
    code += "#include <stdio.h>\n"
    code += "#include <stdlib.h>\n"
    code += "#include <string.h>\n"
    code += "#include <assert.h>\n"
    code += "#include <math.h>\n"
    code += "#include <cblas.h>\n"

    code += "void init();\n"
    code += "void calc(float ** result , float * data);\n"
    code += "int argmax(float * x, int len);\n"

    with open("output/blas_deploy/call_lib.h", "w") as f:
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
    code += indent + "float ** result = (float**)malloc(sizeof(float*)*%d)" \
        ";\n"%(result_count)
    code += "\n"
    # 计算
    code += indent + "// calculation\n"
    code += indent + "calc(result, /* pointer to test data */);\n"
    code += "}\n"

    with open("output/blas_deploy/main.c", "w") as f:
        f.write(code)
    
    # 4. 生成CMakelists.txt
    code = ""
    code += """
cmake_minimum_required(VERSION 3.5)
project(blas_deploy)
set(CMAKE_C_FLAGS "-O3 -Wall -W -pthread")
set(CMAKE_BUILD_TYPE "Release")
include_directories(
    /opt/OpenBLAS/include
)
link_directories(
    /opt/OpenBLAS/lib
)
add_executable(blas_deploy main.c call_lib.c)
target_link_libraries(blas_deploy openblas pthread m)    
"""
    with open("output/blas_deploy/CMakeLists.txt", "w") as f:
        f.write(code)



def gen_block_deploy(
    float_model_dir,
    im2col_shape,
    divided_border,
    sub_matrix_size
):
    '''
    生成用OpenBlas分块计算的代码
    1. 生成计算库, 及其头文件
    2. 生成主函数, 留空给用户填充
    3. 生成CMakeLists
    '''
    calculation_graph = graph.read_calculation_graph(float_model_dir)
    graph.infer_output_shape(calculation_graph)

    # 0. 将权重保存到对应文件夹里  (layer_%d_weight.bin)
    if(not os.path.exists("output/blas_block_deploy/weight")):
        os.system("mkdir -p output/blas_block_deploy/weight")
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or 
           type(node) == op.Dense):
            weight_path = "output/blas_block_deploy/weight/layer_%d_weight.bin"%(n)
            bias_path = "output/blas_block_deploy/weight/layer_%d_bias.bin"%(n)
            node.weight.tofile(weight_path)
            node.bias.tofile(bias_path)
        elif(type(node) == op.Batch_norm2d):
            weight_path = "output/blas_block_deploy/weight/layer_%d_weight.bin"%(n)
            bias_path = "output/blas_block_deploy/weight/layer_%d_bias.bin"%(n)
            running_mean_path = "output/blas_block_deploy/weight/" \
                "layer_%d_running_mean.bin"%(n)
            running_var_path = "output/blas_block_deploy/weight/" \
                "layer_%d_running_var.bin"%(n)
            node.weight.tofile(weight_path)
            node.bias.tofile(bias_path)
            node.running_mean.tofile(running_mean_path)
            node.running_var.tofile(running_var_path)
    
    # 1. 生成计算库
    # 生成头文件调用
    code = ""
    code += "#include \"call_lib.h\""
    code += "\n\n"

    # 生成nn计算函数
    code += gen_blasnn()
    code += "\n\n"

    # 声明权重空间  (layer_%d_weight)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d or 
           type(node) == op.Dense):
            code += "float * layer_%d_weight;\n"%(n)
            code += "float * layer_%d_bias;\n"%(n)
        elif(type(node) == op.Batch_norm2d):
            code += "float * layer_%d_weight;\n"%(n)
            code += "float * layer_%d_bias;\n"%(n)
            code += "float * layer_%d_running_mean;\n"%(n)
            code += "float * layer_%d_running_var;\n"%(n)
    code += "\n\n"

    # 声明分块权重空间 (layer_%d_weight_%d_%d)
    for n, node in enumerate(calculation_graph):
        # 对于conv2d, 需要对权重切块
        if(type(node) != op.Conv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        for i in range(shape_A[0]):
            for j in range(shape_A[1]):
                code += "float * layer_%d_weight_%d_%d;\n"%(n, i, j)
    code += "\n\n"
    
    # 声明计算结果空间  (layer_%d_res, layer_%d_xcol)
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d):
            code += "float * layer_%d_xcol;\n"%(n)
        code += "float * layer_%d_res;\n"%(n)
    code += "\n\n"

    # 声明分块x_col空间
    for n, node in enumerate(calculation_graph):
        # 对于conv2d，需要对x_col切块
        if(type(node) != op.Conv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        for i in range(shape_B[0]):
            for j in range(shape_B[1]):
                code += "float * layer_%d_xcol_%d_%d;\n"%(n, i, j)
    code += "\n\n"

    # 声明分块结果空间
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.Conv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        shape_C = (shape_A[0], shape_B[1])
        for i in range(shape_C[0]):
            for j in range(shape_C[1]):
                code += "float * layer_%d_res_%d_%d;\n"%(n, i, j)
    code += "\n\n"

    # 生成init函数
    code += "void init()\n"
    code += "{\n"
    # 为权重分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d):
            code += "layer_%d_weight = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "layer_%d_bias = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.bias.shape[0])
        elif(type(node) == op.Dense):
            code += "layer_%d_weight = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1])
            code += "layer_%d_bias = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.bias.shape[0])
        elif(type(node) == op.Batch_norm2d):
            code += "layer_%d_weight = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.weight.shape[0])
            code += "layer_%d_bias = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.bias.shape[0])
            code += "layer_%d_running_mean = (float*)malloc(sizeof(float)*%d" \
                ");\n"%(n, node.running_mean.shape[0])
            code += "layer_%d_running_var = (float*)malloc(sizeof(float)*%d" \
                ");\n"%(n, node.running_var.shape[0])
    code += "\n\n"
    # 为分块权重分配空间
    for n, node in enumerate(calculation_graph):
        # 对于conv2d, 需要对权重切块
        if(type(node) != op.Conv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        for i in range(shape_A[0]):
            for j in range(shape_A[1]):
                code += "layer_%d_weight_%d_%d = (float*)malloc(sizeof" \
                    "(float)*%d);\n"%(n, i, j, 
                    sub_matrix_size_A[i][j][0] * sub_matrix_size_A[i][j][1])
    code += "\n\n"
    # 读取权重数据
    code += "FILE * f;\n"
    code += "int l;\n"
    path = os.getcwd()
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d):
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_weight" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(float), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1] * 
                node.weight.shape[2] * node.weight.shape[3])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_bias" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(float), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
        elif(type(node) == op.Dense):
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_weight" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(float), %d, f);\n"%(
                n, node.weight.shape[0] * node.weight.shape[1])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0] * node.weight.shape[1])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_bias" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(float), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
        elif(type(node) == op.Batch_norm2d):
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_weight" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_weight, sizeof(float), %d, f);\n"%(
                n, node.weight.shape[0])
            code += "assert(l == %d);\n"%(
                node.weight.shape[0])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/layer_%d_bias" \
                ".bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_bias, sizeof(float), %d, f);\n"%(
                n, node.bias.shape[0])
            code += "assert(l == %d);\n"%(node.bias.shape[0])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/" \
                "layer_%d_running_mean.bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_running_mean, sizeof(float), %d," \
                " f);\n"%(n, node.running_mean.shape[0])
            code += "assert(l == %d);\n"%(
                node.running_mean.shape[0])
            code += "fclose(f);\n"
            code += "f = fopen(\"%s/output/blas_deploy/weight/" \
                "layer_%d_running_var.bin\", \"rb\");\n"%(path, n)
            code += "l = fread(layer_%d_running_var, sizeof(float), %d," \
                " f);\n"%(n, node.running_var.shape[0])
            code += "assert(l == %d);\n"%(
                node.running_var.shape[0])
            code += "fclose(f);\n"
    code += "\n\n"
    # 将权重数据复制到分块权重中
    code += "float * dst_ptr;\n"
    code += "float * src_ptr;\n"
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.Conv2d):
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
                code += "memcpy(dst_ptr, src_ptr, sizeof(float)*%d);\n"%(
                    copy_end_w - copy_start_w)
                code += "dst_ptr += %d;\n"%(copy_end_w - copy_start_w)
                #   # 如果原始矩阵不够，将右面部分填0，并移动dst_ptr
                if(actual_end_w < end_w):
                    code += "memset(dst_ptr, 0, sizeof(float)*%d);\n"%(
                        end_w - copy_end_w)
                    code += "dst_ptr += %d;\n"%(end_w - copy_end_w)
                #   # 移动src_ptr(下移一行，即加上原始矩阵宽度)
                code += "src_ptr += %d;\n"%(actual_end_w)
                code += "}\n"
                #   # 如果end_h超过了actual_end_h，对剩下的行填0
                if(end_h > actual_end_h):
                    code += "memset(dst_ptr, 0, sizeof(float)*%d);\n"%(
                        (end_h - actual_end_h) * (end_w - start_w))
                code += "\n"
    # 为计算结果分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Conv2d):
            code += "layer_%d_xcol = (float*)malloc(sizeof(float)*%d);\n"%(
                n, node.weight.shape[1] * node.weight.shape[2] * 
                node.weight.shape[3] * node.output_shape[2] * 
                node.output_shape[3])
        length = 1
        for i in node.output_shape:
            length *= i
        code += "layer_%d_res = (float*)malloc(sizeof(float)*%d);\n"%(
            n, length)
    code += "\n\n"
    # 为分块x_col分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.Conv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        for i in range(shape_B[0]):
            for j in range(shape_B[1]):
                code += "layer_%d_xcol_%d_%d = (float*)malloc(sizeof(float)" \
                    "*%d);\n"%(n, i, j, 
                    sub_matrix_size_B[i][j][0] * sub_matrix_size_B[i][j][1])
    code += "\n\n"
    # 为分块结果分配空间
    for n, node in enumerate(calculation_graph):
        if(type(node) != op.Conv2d):
            continue
        conv_index = get_conv2d_index(n, calculation_graph)
        sub_matrix_size_A = sub_matrix_size[conv_index][0]
        sub_matrix_size_B = sub_matrix_size[conv_index][1]
        shape_A = (len(sub_matrix_size_A), len(sub_matrix_size_A[0]))
        shape_B = (len(sub_matrix_size_B), len(sub_matrix_size_B[0]))
        shape_C = (shape_A[0], shape_B[1])
        for i in range(shape_C[0]):
            for j in range(shape_C[1]):
                code += "layer_%d_res_%d_%d = (float*)malloc(sizeof(float)" \
                    "*%d);\n"%(n, i, j, 
                    sub_matrix_size_A[i][0][0] * sub_matrix_size_B[0][j][1])
    code += "\n\n"
    code += "}\n"

    # 生成calc函数
    result_count = 0
    code += "void calc(float ** result, float * data)\n"
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
    code += "float * dst_ptr;\n"
    code += "float * src_ptr;\n"
    for n, node in enumerate(calculation_graph):
        if(type(node) == op.Add or
           type(node) == op.Concat):
            input1_id = node.input1
            input2_id = node.input2
        elif(type(node) == op.Input):
            pass
        else:
            input_id = node.input
        if(type(node) == op.Input):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "input(layer_%d_res, data, %d);\n"%(n, length)
        elif(type(node) == op.Conv2d):
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
            code += "im2col(layer_%d_xcol, layer_%d_res, %d, %d, channels_col, " \
                "height_col, width_col, %d, %d, %d, %d, %d, %d, %d, %d);\n"%(
                n, input_id, 
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3],
                node.weight.shape[2], node.weight.shape[3],
                node.stride[0], node.stride[1], node.padding[0],
                node.padding[1], node.dilation[0], node.dilation[1])
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
                    code += "dst_ptr = layer_%d_xcol_%d_%d;\n"%(n, i, j)
                    code += "src_ptr = &layer_%d_xcol[%d];\n"%(
                        n, copy_start_h * actual_end_w + copy_start_w)
                    #   # 遍历需要copy的每一行
                    code += "for(int i = %d; i<%d; i++) {\n"%(
                        copy_start_h, copy_end_h)
                    #   # copy左侧需要复制的部分，并移动dst_ptr
                    code += "memcpy(dst_ptr, src_ptr, sizeof(float)*%d);\n"%(
                        copy_end_w - copy_start_w)
                    code += "dst_ptr += %d;\n"%(copy_end_w - copy_start_w)
                    #   # 如果原始矩阵不够，将右面部分填0，并移动dst_ptr
                    if(actual_end_w < end_w):
                        code += "memset(dst_ptr, 0, sizeof(float)*%d);\n"%(
                            end_w - copy_end_w)
                        code += "dst_ptr += %d;\n"%(end_w - copy_end_w)
                    #   # 移动src_ptr(下移一行，即加上原始矩阵宽度)
                    code += "src_ptr += %d;\n"%(actual_end_w)
                    code += "}\n"
                    #   # 如果end_h超过了actual_end_h，对剩下的行填0
                    if(end_h > actual_end_h):
                        code += "memset(dst_ptr, 0, sizeof(float)*%d);\n"%(
                            (end_h - actual_end_h) * (end_w - start_w))
                    code += "\n"
            # 分块weight和分块x_col计算，结果存入分块res
            shape_C = (shape_A[0], shape_B[1])
            for i in range(shape_C[0]):
                for j in range(shape_C[1]):
                    code += "memset(layer_%d_res_%d_%d, 0, sizeof(float)*%d)" \
                        ";\n"%(n, i, j, 
                    sub_matrix_size_A[i][0][0] * sub_matrix_size_B[0][j][1])
            M = shape_C[0]
            N = shape_C[1]
            K = shape_A[1]
            for i in range(M):
                for j in range(N):
                    for k in range(K):
                        code += "cblas_sgemm(CblasRowMajor, CblasNoTrans, "\
                            "CblasNoTrans, %d, %d, %d, 1.0f, " \
                            "layer_%d_weight_%d_%d, %d, " \
                            "layer_%d_xcol_%d_%d, %d, 1.0f, " \
                            "layer_%d_res_%d_%d, %d);\n"%(
                            sub_matrix_size_A[i][0][0],
                            sub_matrix_size_B[0][j][1],
                            sub_matrix_size_A[0][k][1],
                            n, i, k, sub_matrix_size_A[0][k][1],
                            n, k, j, sub_matrix_size_B[0][j][1],
                            n, i, j, sub_matrix_size_B[0][j][1]
                            )
            # 分块res复制到res
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
                    code += "dst_ptr = &layer_%d_res[%d];\n"%(
                        n, copy_start_h * actual_end_w + copy_start_w)
                    code += "src_ptr = layer_%d_res_%d_%d;\n"%(n, i, j)
                    #   # 遍历需要copy的每一行
                    code += "for(int i = %d; i<%d; i++) {\n"%(
                        copy_start_h, copy_end_h)
                    #   # copy左侧需要复制的部分，
                    #   #   并移动dst_ptr(下移一行,即加上原始矩阵宽度)
                    code += "memcpy(dst_ptr, src_ptr, sizeof(float)*%d);\n"%(
                        copy_end_w - copy_start_w)
                    code += "dst_ptr += %d;\n"%(actual_end_w)
                    #   # 移动src_ptr(下移一行，即加上块矩阵宽度)
                    code += "src_ptr += %d;\n"%(end_w - start_w)
                    code += "}\n"
                    #   # 对于块矩阵超过原始矩阵的部分，舍去不要
            # 加上bias
            code += "for(int i = 0; i<%d; i++) {\n"%(im2col_shape_A[0])
            code += "for(int j = 0; j<%d; j++) {\n"%(im2col_shape_B[1])
            code += "layer_%d_res[i*%d+j] += layer_%d_bias[i];\n"%(
                n, im2col_shape_B[1], n)
            code += "}\n"
            code += "}\n"
            code += "\n"
        elif(type(node) == op.Maxpool2d):
            code += "maxpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
                "%d, %d, %d, %d, %d, %d, %d, %d);\n"%(n, input_id, 
                calculation_graph[input_id].output_shape[0],
                calculation_graph[input_id].output_shape[1],
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3],
                node.kernel_size[0], node.kernel_size[1],
                node.stride[0], node.stride[1], node.padding[0],
                node.padding[1], node.dilation[0], node.dilation[1])
        elif(type(node) == op.Relu):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "relu(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
        elif(type(node) == op.Flatten):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "flatten(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
        elif(type(node) == op.Dense):
            code += "fc(layer_%d_res, layer_%d_res, layer_%d_weight, " \
                "layer_%d_bias, %d, %d);\n"%(
                n, input_id, n, n, node.output_channel, node.input_channel)
        elif(type(node) == op.Dropout):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "dropout(layer_%d_res, layer_%d_res, %f, %d);\n"%(
                n, input_id, node.p, length)
        elif(type(node) == op.Output):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "output(layer_%d_res, layer_%d_res, %d);\n"%(
                n, input_id, length)
            code += "result[%d] = layer_%d_res;\n"%(result_count, n)
            result_count += 1
        elif(type(node) == op.Add):
            length = 1
            for i in node.output_shape:
                length *= i
            code += "add(layer_%d_res, layer_%d_res, layer_%d_res, %d);\n"%(
                n, input1_id, input2_id, length)
        elif(type(node) == op.Concat):
            # TODO: not supported yet
            raise TypeError("Not supported yet")
        elif(type(node) == op.Avgpool2d):
            code += "avgpool2d(layer_%d_res, layer_%d_res, %d, %d, %d, %d, " \
            "%d, %d, %d, %d, %d, %d);\n"%(n, input_id,
            calculation_graph[input_id].output_shape[0],
            calculation_graph[input_id].output_shape[1],
            calculation_graph[input_id].output_shape[2],
            calculation_graph[input_id].output_shape[3],
            node.kernel_size[0], node.kernel_size[1], node.stride[0],
            node.stride[1], node.padding[0], node.padding[1])
        elif(type(node) == op.Batch_norm2d):
            code += "batch_norm2d(layer_%d_res, layer_%d_res, " \
                "layer_%d_running_mean, layer_%d_running_var, " \
                "layer_%d_weight, layer_%d_bias, %f, %d, %d, %d, %d);\n"%(
                n, input_id, n, n, n, n, node.eps, 
                calculation_graph[input_id].output_shape[0],
                calculation_graph[input_id].output_shape[1],
                calculation_graph[input_id].output_shape[2],
                calculation_graph[input_id].output_shape[3])
        else:
            raise TypeError("Not supported yet")
    
    code += "}\n"
            

    if(not os.path.exists("output/blas_block_deploy")):
        os.system("mkdir -p output/blas_block_deploy")
    with open("output/blas_block_deploy/call_lib.c", "w") as f:
        f.write(code)

    
    # 2. 生成计算库头文件
    code = ""
    code += "#include <stdio.h>\n"
    code += "#include <stdlib.h>\n"
    code += "#include <string.h>\n"
    code += "#include <assert.h>\n"
    code += "#include <math.h>\n"
    code += "#include <cblas.h>\n"

    code += "void init();\n"
    code += "void calc(float ** result , float * data);\n"
    code += "int argmax(float * x, int len);\n"

    with open("output/blas_block_deploy/call_lib.h", "w") as f:
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
    code += indent + "float ** result = (float**)malloc(sizeof(float*)*%d)" \
        ";\n"%(result_count)
    code += "\n"
    # 计算
    code += indent + "// calculation\n"
    code += indent + "calc(result, /* pointer to test data */);\n"
    code += "}\n"

    with open("output/blas_block_deploy/main.c", "w") as f:
        f.write(code)
    
    # 4. 生成CMakelists.txt
    code = ""
    code += """
cmake_minimum_required(VERSION 3.5)
project(blas_block_deploy)
set(CMAKE_C_FLAGS "-O3 -Wall -W -pthread")
set(CMAKE_BUILD_TYPE "Release")
include_directories(
    /opt/OpenBLAS/include
)
link_directories(
    /opt/OpenBLAS/lib
)
add_executable(blas_block_deploy main.c call_lib.c)
target_link_libraries(blas_block_deploy openblas pthread m)    
"""
    with open("output/blas_block_deploy/CMakeLists.txt", "w") as f:
        f.write(code)


def gen_code(
    im2col_shape,
    divided_border,
    submatrix_size,
    calc_process,
    float_model_dir
):
    '''
    生成cpu浮点计算代码, 验证正确性
    '''

    # 生成使用OpenBlas计算的原始代码
    gen_blas_deploy(
        float_model_dir
    )

    # 生成切块计算代码
    gen_block_deploy(
        float_model_dir,
        im2col_shape,
        divided_border,
        submatrix_size
    )