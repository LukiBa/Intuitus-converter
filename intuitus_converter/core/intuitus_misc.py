# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:04:13 2021

@author: lukas
"""

import numpy as np
import json
import pathlib

class addr_cycle():
    def __init__(self,init : int, start: int, end : int): 
        self._start = start
        self._end = end 
        self.position = init
        
    def __call__(self, iterate):
        pos = self.position
        self.position += iterate
        if self.position >= self._end:
            self.position = self.position - self._end + self._start
        return pos

    def reset(self):
        self.position = 0

def reshape_weights(weights,filters,k_size,in_channels,torch_weights_channels_first=True):
    if torch_weights_channels_first:
        assert weights.shape == (filters,in_channels,k_size,k_size), "Dimension missmatch reading weights with shape {} expected {}".format(weights.shape,(filters,in_channels,k_size,k_size))
        resh_weights = np.zeros((k_size,k_size,in_channels,filters))
        for i in range(in_channels):
            for j in range(filters):
                resh_weights[...,i,j] = weights[j,i,...] 
        return resh_weights
    else:
        assert weights.shape == (k_size,k_size,in_channels,filters), "Dimension missmatch reading weights with shape {} expected {}".format(weights.shape,(k_size,k_size,in_channels,filters))
        return weights
        
def convert_torch2qKeras_param(model, weights_path, model_name='yolov4', is_tiny=False, keras2torch_path = './keras_2_torch_names.json',torch_weights_channels_first=True):
    weights_path = pathlib.Path(weights_path).absolute()
    keras2torch_path = pathlib.Path(keras2torch_path).absolute()
    with open(keras2torch_path) as json_file:
        data = json.load(json_file)
        if is_tiny:
            if model_name == 'yolov3':
                layer_size = 13
                tabular = data['yolov3-tiny']
            else:
                layer_size = 21
                tabular = data['yolov4-tiny']
        else:
            if model_name == 'yolov3':
                layer_size = 75
                tabular = data['yolov3']
            else:
                layer_size = 110
                tabular = data['yolov4']

    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i 

        conv_layer = model.get_layer(conv_layer_name)
        
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_channels = conv_layer.input_shape[-1]
        
        weights = np.load(str(weights_path / (tabular[conv_layer_name]+'.weight.npy')))
        weights = reshape_weights(weights,filters,k_size,in_channels,torch_weights_channels_first)
        bias = np.load(str(weights_path / (tabular[conv_layer_name]+'.bias.npy')))
        assert bias.shape == (filters,), "Dimension missmatch reading bias with shape {} expected {}".format(bias.shape,(filters,))
        act_scale = np.load(str(weights_path / (tabular[conv_layer_name]+'.activation_shift.npy')))
        assert act_scale.shape == (1,), "Dimension missmatch reading bias with shape {} expected {}".format(act_scale.shape,(1,))
        act_scale = act_scale[0]
        bias_scale = np.load(str(weights_path / (tabular[conv_layer_name]+'.bias_quant_shift.npy')))
        assert bias_scale.shape == (1,), "Dimension missmatch reading bias with shape {} expected {}".format(bias_scale.shape,(1,))
        bias_scale = bias_scale[0]        
        conv_layer.set_weights([weights, bias])
        conv_layer.out_shift = act_scale
        conv_layer.bias_shift = bias_scale
        conv_layer.quantized_param = True


