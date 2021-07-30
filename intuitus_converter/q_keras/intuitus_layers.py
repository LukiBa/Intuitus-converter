# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:10:28 2020

@author: lukas
"""
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
import tensorflow as tf 

from intuitus_converter.core import float8,float6,MACC_slim,float12,fixed4
import intuitus_converter.misc.util.optain_dataset as load 
import IntuitusExtension as C_impl



def call_numpy (in_tensor,kernel,bias=None, stride = 1):
    in_tensor = np.array(in_tensor)
    kernel = np.array(kernel)
    weights = float6(kernel)
    if type(bias) != None:
        bias = np.array(bias)
        bias_7q7 = np.int16(np.around(2**(8-1)*bias))
    else:
        bias_7q7 = np.zeros(kernel.shape[-1],dtype=np.int16)
    tensor_fl8 = float8(in_tensor)
    print("Conv 2d tensor {}, kernel {}".format(in_tensor.shape,kernel.shape))
    z_exp, z =  C_impl.conv2d_DSP_sim(tensor_fl8.exp,tensor_fl8.mantissa,weights.sign,weights.exp,weights.mantissa,bias_7q7,stride)
    return float8(z_exp,z).to_float32()

def call_numpy_slow(in_tensor,kernel,bias=None,used_padding='same'): 
    in_tensor = np.array(in_tensor)
    kernel = np.array(kernel)
    weights = float6(kernel)
    if type(bias) != type(None):
        bias_in = np.array(bias)
    else:
        bias_in = np.zeros(kernel.shape[-1],dtype=np.int16)
    print("Slow: Conv 2d tensor {}, kernel {}".format(in_tensor.shape,kernel.shape))    
    output_channels = MACC_slim(bias_in)
    if used_padding == 'same':
        tensor = np.zeros([in_tensor.shape[0],in_tensor.shape[1]+2,in_tensor.shape[2]+2,in_tensor.shape[3]])
    tensor[:,1:-1,1:-1,:] = in_tensor  
    tensor_fl8 = float8(tensor)
    for k in range(kernel.shape[0]):
        for l in range(kernel.shape[1]):
            for m in range(in_tensor.shape[-1]):
                    output_channels(tensor_fl8[:,k:tensor.shape[1]-kernel.shape[0]+1+k,l:tensor.shape[2]-kernel.shape[1]+1+l,:],weights[k,l,m,:])  
    print("Done..")
    return np.where(output_channels.to_float32()< 0.0,0.0,output_channels.to_float32())

@keras_export('keras.layers.Conv2D_fl8')     
class Conv2D_fl8(layers.Conv2D): 
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 allow_immediate_maxpool = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.used_padding = padding          
        self.out_shift = None
        self.bias_shift = None
        self.allow_immediate_maxpool = allow_immediate_maxpool
        
        super(Conv2D_fl8, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            trainable=False,
            dynamic=True,
            **kwargs)

    def get_derivative(x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

    def quantize_ufl8(self,inputs): 
        value = tf.math.abs(inputs)
        exp = tf.round((-1)*tf.math.log(value))
        exp = tf.clip_by_value(exp,-8,7)                
        mantissa = tf.round(value*2**(exp+4))
        mantissa = tf.clip_by_value(mantissa,0,15)  
        return exp, mantissa         
    
    def ufl8_to_fl16(self,exps,mantissas):
        return mantissas*2.0**((-1)*exps-4)

    def quantize_fl6(self,inputs): 
        sign = tf.where(inputs<0.0,1.0,0.0)
        value = tf.math.abs(inputs)
        exp = tf.round((-1)*tf.math.log(value))
        exp = tf.clip_by_value(exp,-4,3)                
        mantissa = tf.round(value*2**(exp+3))
        mantissa = tf.clip_by_value(mantissa,0,7)  
        return sign,exp, mantissa         
    
    def fl6_to_fl16(self,signs,exps,mantissas):
        return tf.where(signs==0.0,mantissas*2.0**((-1)*exps-3),(-1)*mantissas*2.0**((-1)*exps-3))
    
    def build(self, input_shape):
        super().build(input_shape)
        
    def __call__(self, *args, **kwargs):
         return super().__call__(*args, **kwargs)
    
    def call(self, inputs):
        # Check if the input_shape in call() is different from that in build().
        # If they are different, recreate the _convolution_op to avoid the stateful
        # behavior.
        # call_input_shape = inputs.get_shape()
        # recreate_conv_op = (
        #     call_input_shape[1:] != self._build_conv_op_input_shape[1:])
    
        # if recreate_conv_op:
        #     self._convolution_op = nn_ops.Convolution(
        #         call_input_shape,
        #         filter_shape=self.kernel.shape,
        #         dilation_rate=self.dilation_rate,
        #         strides=self.strides,
        #         padding=self._padding_op,
        #         data_format=self._conv_op_data_format)
    
        # # Apply causal padding to inputs for Conv1D.
        # if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
        #     inputs = array_ops.pad(inputs, self._compute_causal_padding())
            
        # exps, mantissas = self.quantize_ufl8(inputs)
        # inputs_fl8 = self.ufl8_to_fl16(exps, mantissas)
        # k_signs, k_exps, k_mantissas = self.quantize_fl6(self.kernel)
        # kernels_fl6 = self.fl6_to_fl16(k_signs, k_exps, k_mantissas)
        
        # outputs = self._convolution_op(inputs_fl8, kernels_fl6)
    
        # if self.use_bias:
        #   if self.data_format == 'channels_first':
        #     if self.rank == 1:
        #       # nn.bias_add does not accept a 1D input tensor.
        #       bias = array_ops.reshape(self.bias, (1, self.filters, 1))
        #       outputs += bias
        #     else:
        #       outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        #   else:
        #     outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')
    
        # if self.activation is not None:
        #   return self.activation(outputs)
        # return outputs        
        
        
        
        if self.data_format == 'channels_first':
            raise NotImplementedError("Channels first not implemented")
        call_input_shape = inputs.get_shape()
        recreate_conv_op = (
            call_input_shape[1:] != self._build_conv_op_input_shape[1:])
          
        if recreate_conv_op:
          self._convolution_op = nn_ops.Convolution(
              call_input_shape,
              filter_shape=self.kernel.shape,
              dilation_rate=self.dilation_rate,
              strides=self.strides,
              padding=self._padding_op,
              data_format=self._conv_op_data_format)
          
        # Apply causal padding to inputs for Conv1D.
        if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
          inputs = array_ops.pad(inputs, self._compute_causal_padding())
        
        outputs = tf.numpy_function(call_numpy, [inputs,self.kernel,self.bias] ,tf.float16)
        return outputs
            
    def sim_hw(self,in_tensor): 
        weight_list = self.get_weights()
        strides=self.strides
        weights = float6(weight_list[0])
        if self.use_bias == True:
            bias = weight_list[1]
        else:
            bias = np.zeros([self.filters])
            
        output_channels = np.array([MACC_slim(bias[i]) for i in range(self.filters)])    
        if self.used_padding == 'same':
            if strides == (2,2):
                pad = 1
            else:
                pad = 2            
            if isinstance(in_tensor,float8):
                tensor = np.zeros([in_tensor.shape[0],in_tensor.shape[1]+pad,in_tensor.shape[2]+pad,in_tensor.shape[3]])
                tensor_fl8 = float8(tensor)
                tensor_fl8[:,pad-1:-1,pad-1:-1,:] = in_tensor
            else:
                tensor = np.zeros([in_tensor.shape[0],in_tensor.shape[1]+pad,in_tensor.shape[2]+pad,in_tensor.shape[3]])
                tensor[:,pad-1:-1,pad-1:-1,:] = in_tensor  
                tensor_fl8 = float8(tensor)
        else:
            if isinstance(in_tensor,float8):
                tensor_fl8 = in_tensor
            else:
                tensor_fl8 = float8(in_tensor)
            
        print("Conv 2d tensor {}, kernel {}".format(in_tensor.shape,weights.shape))
        print("channels last")
        for m in range(in_tensor.shape[-1]): # input channels
            for k in range(weights.shape[0]):
                for l in range(weights.shape[1]):
                    for n in range(self.filters):
                        output_channels[n](tensor_fl8[:,k:tensor.shape[1]-weights.shape[0]+1+k:strides[0],l:tensor.shape[2]-weights.shape[1]+1+l:strides[1],m],weights[k,l,m,n])  
        
        #for n in range(self.filters):
        #    output_channels[n].normalize()
        
        print("Done..")
        exp = np.stack([output_channels[i].exp for i in range(self.filters)],axis=3)
        mantissa = np.stack([output_channels[i].mantissa for i in range(self.filters)],axis=3)
        exp = np.where(mantissa<0,7,exp)
        mantissa = np.where(mantissa<0,0,mantissa)
        mantissa = np.right_shift(mantissa,4)
        res = float8(exp,mantissa)
        res.normalize()
        return res

    def sim_hw_in_C (self,in_tensor):
        strides=self.strides
        if strides == (1,1):
            stride = 1
        elif strides == (2,2):
            stride = 2
        else:
            raise NotImplementedError('Stride is not supported yet. Edit ccconv.c in IntuitusExtention to add more strides.')  
        weight_list = self.get_weights()
        weights = float6(weight_list[0])
        if self.use_bias == True:
            bias = weight_list[1]
        else:
            bias = np.zeros([self.filters],dtype=np.int16)
            
        if self.used_padding != 'same':
            raise NotImplementedError('Until now only same padding is implemented')  
        tensor_fl8 = float8(in_tensor)
        bias_fi4 = fixed4(bias)
        exp_shift = np.zeros(bias.shape,dtype=np.int8)
        print("Conv 2d tensor {}, kernel {}".format(in_tensor.shape,weights.shape))
        exp,mantissa =  C_impl.conv2d_fpga(tensor_fl8.exp,tensor_fl8.mantissa.astype(np.int8),
                                       weights.exp,weights.signed_mantissa,
                                       bias_fi4.exp, bias_fi4.signed_mantissa,
                                       exp_shift,stride,0)
        res = float8(exp,mantissa)
        #res.normalize()
        return res
        
@keras_export('keras.layers.Conv2D_int8')            
class Conv2D_int8(layers.Conv2D): 
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_bias=True,
                 allow_immediate_maxpool = True,
                 outshift = None,
                 biasshift = None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.used_padding = padding 
        self.out_shift = outshift  
        self.bias_shift = biasshift
        self.quantized_param = False
        self.allow_immediate_maxpool = allow_immediate_maxpool
        
        super(Conv2D_int8, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            #groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
            
    def sim_hw(self,in_tensor): 
        weight_list = self.get_weights()
        stride=self.strides
        out_shift = self.out_shift
        if not self.quantized_param:
            weight = np.clip(np.round(weight_list[0]*2**5),(-1)*(2**(5-1)-1),2**(5-1)-1)
        else:
            weight = weight_list[0]
        out_shape = (in_tensor.shape[0],int(in_tensor.shape[1]/stride[0]),int(in_tensor.shape[2]/stride[1]),self.filters)
            
        if self.used_padding == 'same':
            padding_h = int(weight.shape[0]/2.0)
            padding_v = int(weight.shape[1]/2.0)
            padding_t = int(padding_h - (stride[0]-1.0))
            padding_b = int(padding_h)
            padding_l = int(padding_v - (stride[1]-1.0))
            padding_r = int(padding_v)
            
            tensor = np.zeros((in_tensor.shape[0],in_tensor.shape[1]+padding_t+padding_b,
                               in_tensor.shape[2]+padding_l+padding_r,in_tensor.shape[3]))
            tensor[:,padding_t:tensor.shape[1]-padding_b,padding_l:tensor.shape[2]-padding_r,:] = in_tensor

        if self.use_bias == True:
            if not self.quantized_param:
                bias = weight_list[1]
                bias = bias * 2**5
                bias = np.clip(bias,(-1) * 2**5, 2**5-1)
                bias = np.round(bias)  
                bias = bias * 2**(14-6-out_shift)   
            else:
                bias = weight_list[1] * 2**(15-6-self.bias_shift)  
            if weight.shape[-1] != bias.shape[0]:
                raise IndexError("Dimension Missmatch of weights and bias")   
            
            output = np.ones(out_shape) * bias.reshape(1, 1, 1, -1)
            
        else:
            output = np.zeros(out_shape)
         
        o_h = output.shape[1] * stride[0]
        o_w = output.shape[2] * stride[1]
        overflows = 0
        #out_shift -= 1
        interchannel_shift = np.clip(out_shift,0,4)
        
        for c in range(tensor.shape[-1]): # in channels 
            for h in range(weight.shape[0]):
                for w in range(weight.shape[1]):
                    output += tensor[:,h:o_h+h:stride[0],w:o_w+w:stride[1],c:c+1]*weight[h,w,c,:].reshape(1,1,1,-1)
                    overflows += output[output>(2.0**(15-1.0)-1.0)].size
                    
            output = output * 2**(-3.0+interchannel_shift)
            output = np.floor(output)
            output = np.clip(output,-1.0*2.0**(12-1.0), 2.0**(12-1.0)-1.0)
            output = output * 2**(3.0-interchannel_shift)
        
        output = output * 2**(-3.0+out_shift)
        output = np.floor(output * 2**(8-12))
        if self.activation == tf.keras.activations.relu:
            output = np.where(output<0.0,0.0,output)     
        print("Overflows: {}".format(overflows))               
          
        return output
