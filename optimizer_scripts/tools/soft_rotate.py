import onnx.helper
import numpy as np
import logging
from . import helper
from . import other
from . import modhelper

def value_info_transpose(g, value_info):
    if value_info is None:
        return
    if len(value_info.type.tensor_type.shape.dim) != 4:
        return
    temp = value_info.type.tensor_type.shape.dim[2].dim_value
    value_info.type.tensor_type.shape.dim[2].dim_value = value_info.type.tensor_type.shape.dim[3].dim_value
    value_info.type.tensor_type.shape.dim[3].dim_value = temp

def conv_weight_rotate(g, const_node, counter_clock=False):
    # Transpose
    old_weight_np = helper.constant_to_numpy(const_node)
    if counter_clock:
        new_weight_np = np.rot90(old_weight_np, 3, (3, 2))
    else:
        new_weight_np = np.rot90(old_weight_np, 1, (3, 2))
    # Create new node
    new_node = helper.numpy_to_constant(const_node.output[0], new_weight_np)
    g.node.remove(const_node)
    g.node.extend([new_node])

def soft_rotate(m, degree):
    '''Transpose the input without adding any Transpose layers.
    '''
    g = m.graph
    # Check argument
    if degree != 90 and degree != 270:
        print("Currently, rotation only suppot 90 or 270 degree clockwise.")
        quit(1)
    # Transpose input
    for input_value in g.input:
        value_info_transpose(g, input_value)
    # Transpose output
    for output_value in g.output:
        value_info_transpose(g, output_value)
    last_shape = None
    for node in g.node:
        if node.op_type == 'Conv':
            # Transpose attributes
            dilations = helper.get_attribute_by_name(node, 'dilations')
            if dilations != None:
                temp = dilations.ints[0]
                dilations.ints[0] = dilations.ints[1]
                dilations.ints[1] = temp
            kernel_shape = helper.get_attribute_by_name(node, 'kernel_shape')
            if kernel_shape != None:
                temp = kernel_shape.ints[0]
                kernel_shape.ints[0] = kernel_shape.ints[1]
                kernel_shape.ints[1] = temp
            pads = helper.get_attribute_by_name(node, 'pads')
            if pads != None:
                if degree == 90:
                    temp = pads.ints[0]
                    pads.ints[0] = pads.ints[3]
                    pads.ints[3] = pads.ints[2]
                    pads.ints[2] = pads.ints[1]
                    pads.ints[1] = temp
                elif degree == 270:
                    temp = pads.ints[0]
                    pads.ints[0] = pads.ints[1]
                    pads.ints[1] = pads.ints[2]
                    pads.ints[2] = pads.ints[3]
                    pads.ints[3] = temp
            strides = helper.get_attribute_by_name(node, 'strides')
            if strides != None:
                temp = strides.ints[0]
                strides.ints[0] = strides.ints[1]
                strides.ints[1] = temp
            # Transpose weight
            conv_weight_rotate(g, helper.find_node_by_output_name(g, node.input[1]), degree == 270)
        elif node.op_type == 'ConvTranspose':
            # Transpose attributes
            dilations = helper.get_attribute_by_name(node, 'dilations')
            if dilations != None:
                temp = dilations.ints[0]
                dilations.ints[0] = dilations.ints[1]
                dilations.ints[1] = temp
            kernel_shape = helper.get_attribute_by_name(node, 'kernel_shape')
            if kernel_shape != None:
                temp = kernel_shape.ints[0]
                kernel_shape.ints[0] = kernel_shape.ints[1]
                kernel_shape.ints[1] = temp
            output_padding = helper.get_attribute_by_name(node, 'output_padding')
            if output_padding != None:
                temp = output_padding.ints[0]
                output_padding.ints[0] = output_padding.ints[1]
                output_padding.ints[1] = temp
            output_shape = helper.get_attribute_by_name(node, 'output_shape')
            if output_shape != None:
                temp = output_shape.ints[2]
                output_shape.ints[2] = output_shape.ints[3]
                output_shape.ints[3] = temp
            pads = helper.get_attribute_by_name(node, 'pads')
            if pads != None:
                if degree == 90:
                    temp = pads.ints[0]
                    pads.ints[0] = pads.ints[3]
                    pads.ints[3] = pads.ints[2]
                    pads.ints[2] = pads.ints[1]
                    pads.ints[1] = temp
                elif degree == 270:
                    temp = pads.ints[0]
                    pads.ints[0] = pads.ints[1]
                    pads.ints[1] = pads.ints[2]
                    pads.ints[2] = pads.ints[3]
                    pads.ints[3] = temp
            strides = helper.get_attribute_by_name(node, 'strides')
            if strides != None:
                temp = strides.ints[0]
                strides.ints[0] = strides.ints[1]
                strides.ints[1] = temp
            conv_weight_rotate(g, helper.find_node_by_output_name(g, node.input[1]), degree == 270)
        elif node.op_type == 'Pad':
            # Transpose attributes
            pads = helper.get_attribute_by_name(node, 'pads')
            if degree == 90:
                temp = pads.ints[0]
                pads.ints[0] = pads.ints[3]
                pads.ints[3] = pads.ints[2]
                pads.ints[2] = pads.ints[1]
                pads.ints[1] = temp
            elif degree == 270:
                temp = pads.ints[0]
                pads.ints[0] = pads.ints[1]
                pads.ints[1] = pads.ints[2]
                pads.ints[2] = pads.ints[3]
                pads.ints[3] = temp
            # Transpose weight
        elif node.op_type == 'Flatten':
            shape = helper.get_shape_from_value_info(helper.find_value_by_name(g, node.input[0]))
            if len(shape) == 4:
                last_shape = shape
        elif node.op_type == 'Gemm':
            # Check previous
            if len(last_shape) != 4:
                continue
            input_shape = helper.get_shape_from_value_info(helper.find_value_by_name(g, node.input[0]))
            if last_shape[0] * last_shape[1] * last_shape[2] * last_shape[3] != input_shape[0] * input_shape[1]:
                continue
            # Get const_node
            const_node = helper.find_node_by_output_name(g, node.input[1])
            # Transpose
            old_weight_np = helper.constant_to_numpy(const_node)
            old_weight_np = np.reshape(old_weight_np, (last_shape[0], last_shape[1], last_shape[2], last_shape[3], -1))
            if degree == 90:
                new_weight_np = np.rot90(old_weight_np, 1, (3, 2))
            else:
                new_weight_np = np.rot90(old_weight_np, 3, (3, 2))
            new_weight_np = np.reshape(new_weight_np, (-1, new_weight_np.shape[-1]))
            # Create new node
            new_node = helper.numpy_to_constant(const_node.output[0], new_weight_np)
            g.node.remove(const_node)
            g.node.extend([new_node])
        elif node.op_type == 'AveragePool':
            # Transpose attributes
            kernel_shape = helper.get_attribute_by_name(node, 'kernel_shape')
            if kernel_shape != None:
                temp = kernel_shape.ints[0]
                kernel_shape.ints[0] = kernel_shape.ints[1]
                kernel_shape.ints[1] = temp
            pads = helper.get_attribute_by_name(node, 'pads')
            if pads != None:
                if degree == 90:
                    temp = pads.ints[0]
                    pads.ints[0] = pads.ints[3]
                    pads.ints[3] = pads.ints[2]
                    pads.ints[2] = pads.ints[1]
                    pads.ints[1] = temp
                elif degree == 270:
                    temp = pads.ints[0]
                    pads.ints[0] = pads.ints[1]
                    pads.ints[1] = pads.ints[2]
                    pads.ints[2] = pads.ints[3]
                    pads.ints[3] = temp
            strides = helper.get_attribute_by_name(node, 'strides')
            if strides != None:
                temp = strides.ints[0]
                strides.ints[0] = strides.ints[1]
                strides.ints[1] = temp
        elif node.op_type == 'MaxPool':
            # Transpose attributes
            kernel_shape = helper.get_attribute_by_name(node, 'kernel_shape')
            if kernel_shape != None:
                temp = kernel_shape.ints[0]
                kernel_shape.ints[0] = kernel_shape.ints[1]
                kernel_shape.ints[1] = temp
            pads = helper.get_attribute_by_name(node, 'pads')
            if pads != None:
                if degree == 90:
                    temp = pads.ints[0]
                    pads.ints[0] = pads.ints[3]
                    pads.ints[3] = pads.ints[2]
                    pads.ints[2] = pads.ints[1]
                    pads.ints[1] = temp
                elif degree == 270:
                    temp = pads.ints[0]
                    pads.ints[0] = pads.ints[1]
                    pads.ints[1] = pads.ints[2]
                    pads.ints[2] = pads.ints[3]
                    pads.ints[3] = temp
            strides = helper.get_attribute_by_name(node, 'strides')
            if strides != None:
                temp = strides.ints[0]
                strides.ints[0] = strides.ints[1]
                strides.ints[1] = temp
        elif node.op_type == 'Reshape':
            logging.warning("Reshape may cause problems in soft tranpose.")
        elif node.op_type == 'Transpose':
            logging.warning("Transpose may cause problems in soft tranpose.")
        elif node.op_type == 'Squeeze':
            logging.warning("Squeeze may cause problems in soft tranpose.")
        elif node.op_type == 'Unsqueeze':
            logging.warning("Unsqueeze may cause problems in soft tranpose.")
    other.topological_sort(g)
    while len(g.value_info) != 0:
        g.value_info.pop()
    return modhelper.inference_shapes(m)
