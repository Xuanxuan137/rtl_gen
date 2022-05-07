
import op

def infer_im2col_shape(calculation_graph):
    '''
    推算计算图中卷积算子im2col后的矩阵边长
    '''
    im2col_shape = []
    for node in calculation_graph:
        if(type(node) != op.QConv2d):
            continue
        output_channel = node.output_channel
        input_channel = node.input_channel
        kernel_size = node.kernel_size
        stride = node.stride
        padding = node.padding
        dilation = node.dilation
        output_shape = node.output_shape
        weight_shape = [output_channel, input_channel, kernel_size[0], kernel_size[1]]
        bias_shape = [output_channel]
        
        weight_matrix_row = weight_shape[0]
        weight_matrix_col = weight_shape[1] * weight_shape[2] * weight_shape[3]
        
        feature_map_matrix_row = weight_shape[1] * weight_shape[2] * weight_shape[3]
        feature_map_matrix_col = output_shape[2] * output_shape[3]
        
        im2col_shape.append(([weight_matrix_row, weight_matrix_col], 
            [feature_map_matrix_row, feature_map_matrix_col]))


    return im2col_shape