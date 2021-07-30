# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:27:50 2020

@author: lukas
"""
import pathlib
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf

from intuitus_converter.core import Command,Intuitus_Basis, Fmap_Command_interpreter, Weight_Command_interprter, addr_cycle, float6,vectorize_layer_settings
from intuitus_converter.q_keras import Conv2D_fl8, Conv2D_int8

"""
Possible future features:
    - Each class inherits from the corresponding keras class. 
        - Enables training 
        - Enables to use all Keras functions: (model import, tensorboard etc.)

"""

class Intuitus_Converter(Intuitus_Basis): 
    def __init__(self, model : models.Model, out_path : pathlib.Path = './', HyperPar_Path = None,use_float8=False,pre_quantized=False):
        if HyperPar_Path == None:
            if use_float8:
                HyperPar_Path = pathlib.Path(__file__).absolute().parents[0] / "Intuitus_HyperPar.json"
            else:
                HyperPar_Path = pathlib.Path(__file__).absolute().parents[0] / "Intuitus_HyperPar_int8.json"
        super(Intuitus_Converter,self).__init__(HyperPar_Path)
        self.keras_model = model
        self._Fmap_translator = Fmap_Command_interpreter(self.HyperPar_Path)    
        self._Weight_translator = Weight_Command_interprter(self.HyperPar_Path)   
        self.out_path = out_path
        self.use_float8 = use_float8
        self.pre_quantized = pre_quantized
        
    
    def translate(self):
        """
        Translates model structure to an executealbe list of commands. 
        Result is a list of commands (array) which includes all informations 
        the intuitus driver needs to execute the neural network structure 

        """
        max_pooling = 0
        
        layer_list = self.keras_model.layers
        self.layer_command_list = []
        weight_address_init = 0
        first_layer = 1
        
        for i in range(len(layer_list)) :
            if isinstance(layer_list[i],layers.Conv2D) or isinstance(layer_list[i],Conv2D_fl8) or isinstance(layer_list[i],Conv2D_int8): 
                max_pooling = 0
                if i+1 < len(layer_list) and isinstance(layer_list[i+1],layers.MaxPooling2D):
                   if layer_list[i+1].strides == (2,2) and layer_list[i+1].pool_size == (2,2):
                       max_pooling = 1 
                   else: 
                       print(Warning("Warning: " + layer_list[i+1].name + " with Stride 1 not supported by FPGA. Use Software implementation."))
                        
                self.layer_command_list.append(self.translate_conv2d(layer_list[i],i,max_pooling,
                                                                     weight_address_init,first_layer))  
                weight_address_init = self.layer_command_list[-1][-1] #+ layer_list[i].kernel_size[0]*layer_list[i].kernel_size[1]*self.layer_command_list[-1][0][-1].get_iterations()
                
            elif isinstance(layer_list[i],layers.DepthwiseConv2D):
                if isinstance(layer_list[i-1],layers.Conv2D) and layer_list[i-1].kernel_size == (1,1):
                    self.layer_command_list.append(self.translate_inversebottleneck(layer_list[i],layer_list[i-1])) 
                    i +=1
                else:
                    raise NotImplementedError
                max_pooling = 0
            elif isinstance(layer_list[i],layers.MaxPooling2D):
                if max_pooling == 0:
                    print(Warning("Warning: " + layer_list[i].name + " not translated."))
                continue
            else: 
                print(Warning("Warning: " + layer_list[i].name + " not translated."))
                continue
            
            self.save_commands(self.layer_command_list[-1],layer_list[i].name)
            first_layer = 0

 
                
        self.translate_exec = 1
        return self.layer_command_list

    def translate_layer(self,layer_nbr,max_pooling=0,first_layer=1):
        """
        Translates a single layer to an executeable list of commands. 
        Result is a list of commands (array) which includes all informations 
        the intuitus driver needs to execute the neural network structure 

        """
        layer = self.keras_model.layers[layer_nbr]
        if isinstance(layer,layers.Conv2D): 
            commands = self.translate_conv2d(layer,layer_nbr,max_pooling=max_pooling,first_layer=first_layer)
        else: 
            raise NotImplementedError   
        
        self.save_commands(commands,layer.name)
        return commands
    
    def save_commands(self,commands,name):          
        self.out_path.mkdir(parents=True, exist_ok=True)
        command_lengths = np.array([i.size for i in commands[1]],dtype=np.uint32)
        np.savez_compressed(str(self.out_path / name), tx_bin=np.concatenate(commands[1]), tx_com_len=command_lengths, tx_tile=commands[2], tx_info=commands[3], 
                                                            rx_tile=commands[4])    
        
        
    def export_command_list(self):
        """
        Exports the translated command list. 

        """
        if self.translate_exec == 0: 
            return self.translate()
        else:
            return self.layer_command_list
        
    def translate_conv2d(self,layer : layers.Conv2D, layer_idx: int, max_pooling = 0, weight_addr_init=0, first_layer=0):
        """
        Creates a command list for the Intuitus hw accelerator based on the 
        properties of a Keras conv2d layer class 

        Parameters
        ----------
        layer : tensorflow.python.keras.layers.convolutional.Conv2D
            Keras Conv2d layer.
        layer_idx : int
            Layer index 
        weight_addr_init : int
            Position of weight address of previous layer 

        Raises
        ------
        NotImplementedError
            Since Keras offers a higher flexibility than Intuitus the range 
            of the kernel_size and strides is restiricted.

        Returns
        -------
        command_list :  np.array [Th,Tw,Ci,Trld]
            Th... Tile number in height axis  
            Tw... Tile number in width axis 
            Ci... Input channel number 
            Trld... Tile reload number. (Depends on output channel number, Image shape and Intuitus hyperparameter)
            Array of commands for each image tile of a layer input. 

        """
        input_shape = layer.input_shape[1:]
        in_channels = input_shape[2]
        kernel_size = layer.kernel_size
        strides = layer.strides
        use_bias = layer.use_bias        
        weights_bias = layer.get_weights()
        if not layer.allow_immediate_maxpool and max_pooling != 0:
            print(Warning("Warning: Immediate maxpooling inside FPGA is not allowed for layer: " + layer.name + ". Use Software implementation instead. Reason can be that conv2d output is required anywhere else in network."))
            max_pooling = 0
        
        if layer.filters%self.syst_array_struct['SYST_ARRAY_WIDTH'] != 0:
            print(Warning("Warning: " + layer.name + ": using {} filter is not optimal for FPGA. Think about using a multiple of {} filters. Weight tensor is extended with additonal zero filters anyhow.".format(layer.filters,self.syst_array_struct['SYST_ARRAY_WIDTH'])))
            filters = int(np.ceil(layer.filters/self.syst_array_struct['SYST_ARRAY_WIDTH'])*self.syst_array_struct['SYST_ARRAY_WIDTH'])
            weights = np.zeros(weights_bias[0].shape[:3]+(filters,))
            weights[...,:layer.filters] = weights_bias[0]
            if use_bias:
                bias = np.zeros((filters))
                bias[:layer.filters] = weights_bias[1]
        else:
            filters = layer.filters
            weights = weights_bias[0]
            if use_bias:
                bias = weights_bias[1]
        
        
        out_channels = layer.filters
        if layer.out_shift == None:
            raise Exception("layer " + layer.name + " requires valid outshift value.")
        
        if kernel_size == (1,1): 
            op_mode = "Conv1x1"
        elif kernel_size == (3,3): 
            op_mode = "Conv3x3"
        elif kernel_size == (5,5): 
            op_mode = "Conv5x5"
        else:
            raise NotImplementedError("Only 1x1, 3x3 and 5x5 convolutions are supportet yet. Set kernel size ether to (1,1), (3,3) or (5,5).")
        if strides == (1,1): 
            stride = 1
        elif strides == (2,2):
            stride = 2
        else: 
            raise NotImplementedError("Only symetrical strides of ether 1 or 2 are supported. Set strides ether to (1,1), (2,2).")

        if self.pu_mode == "float8":
            if use_bias:
                q_bias = self.bias_to_fixed4(bias)
                q_bias = q_bias.to_binary()
            q_weights = float6(weights).to_binary()
            dsp_feedback_len = 3
        elif self.pu_mode == "int8":
            if layer.quantized_param:
                q_bias = np.int8(bias)
                w_sign = np.zeros(weights.shape,dtype=np.int8)
                w_sign = np.where(weights<0.0,1<<self.quantization['WEIGHT_MANTISSA_WIDTH'],w_sign)
                q_weights = np.int8(np.abs(weights))
                q_weights = np.bitwise_or(q_weights,w_sign)
            else:
                if use_bias:
                    q_bias = np.int8(np.clip(np.round(bias*2**(self.quantization['BIAS_WIDTH']-1)),
                                             (-1) *2**(self.quantization['BIAS_WIDTH']-1),
                                             2**(self.quantization['BIAS_WIDTH']-1)-1))
                    
                    
                # weight uses 1th complement 
                w_sign = np.zeros(weights.shape,dtype=np.int8)
                w_sign = np.where(weights<0.0,1<<self.quantization['WEIGHT_MANTISSA_WIDTH'],w_sign)
                q_weights = np.int8(np.clip(np.round(np.abs(weights)*2**(self.quantization['WEIGHT_MANTISSA_WIDTH'])),
                                            0,2**(self.quantization['WEIGHT_MANTISSA_WIDTH'])-1))
                q_weights = np.bitwise_or(q_weights,w_sign)
            dsp_feedback_len = 1
        else:
            raise NotImplementedError("Processing unit mode: " + self.pu_mode + " not supported yet.")

        tile_heights = self._get_tile_height(input_shape[0],kernel_size[0],stride,max_pooling)
        tile_widths = self._get_tile_width(input_shape[1],kernel_size[0],stride,max_pooling,dsp_feedback_len)
        tile_size_in = [tile_heights[0], tile_widths[0]]
        tile_size_out = [tile_heights[1], tile_widths[1]]  
        tile_size_pe = np.array([np.array(tile_heights[1]), np.array(tile_widths[1])]) 
        if max_pooling:
            tile_size_pe *= 2
        tile_number = (len(tile_size_in[0]), len(tile_size_in[1]))
        iterations, tile_filters, tile_recurrence, tile_loads, weight_lengths, weight_widths = self._get_tile_loads(filters,tile_size_pe,kernel_size,use_bias)
        
     
        # Create command for each tile loaded into hardware accelerator 
        command_array = np.ndarray([np.max(tile_loads),tile_number[0],tile_number[1]],object)
        for i in range(np.max(tile_loads)):
            for j in range(tile_number[0]):
                for k in range(tile_number[1]):
                    if i >= tile_loads[j,k]:
                        command_array[i,j,k] = Command(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) # Dummy --> deleted afterwards
                        continue
                    command_array[i,j,k] = Command(op_mode,stride,0,tile_size_in[0][0],tile_size_in[1][0],
                                                  0,iterations[j,k][i], tile_recurrence[j,k],
                                                  tile_loads[j,k],tile_size_out[0][0],tile_size_out[1][0],
                                                  tile_filters[j,k][i],in_channels,weight_lengths[j,k][i],
                                                  weight_widths[j,k][i],max_pooling,1,self.quantization['BIAS_WIDTH'],
                                                  0)
              
        if first_layer != 0:
            command_array[0,0,0].set_reset_weight_cache(1)
        set_tile_height_vec = np.vectorize(Command.set_tile_height,otypes=[object])
        set_tile_width_vec = np.vectorize(Command.set_tile_width,otypes=[object])
        set_tile_height_out_vec = np.vectorize(Command.set_output_tile_height,otypes=[object])
        set_tile_width_out_vec = np.vectorize(Command.set_output_tile_width,otypes=[object])        
        for i in range(tile_number[0]):
            set_tile_height_vec(command_array[:,i,:],tile_size_in[0][i])
            set_tile_height_out_vec(command_array[:,i,:],tile_size_out[0][i])
        for i in range(tile_number[1]):
            set_tile_width_vec(command_array[:,:,i],tile_size_in[1][i])    
            set_tile_width_out_vec(command_array[:,:,i],tile_size_out[1][i]) 
            
        if kernel_size[0] > 3 or kernel_size[0] == 3 and stride == 1:
            add_padding_vec = np.vectorize(Command.add_padding,otypes=[object])
            add_padding_vec(command_array[:,0,:],1) # "0001" Upper padding
            add_padding_vec(command_array[:,:,0],2) # "0010" Left padding 
            add_padding_vec(command_array[:,:,-1],4) # "0100" Right padding         
            add_padding_vec(command_array[:,-1,:],8) # "1000" Lower padding
            add_padding_vec(command_array[:,0,0],3) # "0011" Upper and Left padding
            add_padding_vec(command_array[:,0,-1],5) # "0101" Upper and right padding 
            add_padding_vec(command_array[:,-1,0],10) # "1010"  Lower and left padding
            add_padding_vec(command_array[:,-1,-1],12) # "1100" Lower and right padding 
        elif kernel_size[0] == 3:
            add_padding_vec = np.vectorize(Command.add_padding,otypes=[object])
            add_padding_vec(command_array[:,-1],4) # "0100" Right padding         
            add_padding_vec(command_array[:,-1,:],8) # "1000" Lower padding
            add_padding_vec(command_array[:,-1,-1],12) # "1100" Lower and right padding 
        
        overlays_right = np.int32(np.floor(kernel_size[0]/2))
        if stride == 1:
            overlays_left = overlays_right
        else:
            overlays_left = overlays_right-1
            if input_shape[1]%2 ==1:  # For odd image width --> No common practice to use stride 2 with odd image size
                raise NotImplementedError("Using stride 2 for odd image size is not supported by hardware.")
                overlays_right -= 1
 
        
        for i in range(np.max(tile_loads)):
            if kernel_size[0]*kernel_size[1]*iterations[0,0][i]*in_channels <= self.cache_sizes["Weight cache block size"]:
                command_array[i,0,0].set_new_weights(1) # reuse weights if weight cache is sufficient
            else:
                set_new_weights_vec = np.vectorize(Command.set_new_weights,otypes=[object])
                set_new_weights_vec(command_array[i,:,:],1)
                        
        weight_address_iter = addr_cycle(weight_addr_init,0,self.cache_sizes["Weight cache block size"]) 
    
       
        command_binary_array = np.ndarray([np.max(tile_loads),tile_number[0],tile_number[1]],object)
        in_tiles_array = np.ndarray([np.max(tile_loads),tile_number[0],tile_number[1]],object)
        in_tile_info_array = np.ndarray([np.max(tile_loads),tile_number[0],tile_number[1]],object)
        out_tiles_array = np.ndarray([np.max(tile_loads),tile_number[0],tile_number[1]],object)
        weight_info_array = np.ndarray([np.max(tile_loads),tile_number[0],tile_number[1]],object)
            
        tile_h = overlays_left +overlays_right
        tile_h_o = 0
        th = 0
        th_o = 0
        
        for i in range(command_array.shape[1]):
            th += tile_h - overlays_left -overlays_right
            th_o += tile_h_o
            tw = 0
            tw_o = 0
            tile_w = overlays_left +overlays_right
            tile_w_o = 0
            for j in range(command_array.shape[2]):
 
                tw += tile_w - overlays_left -overlays_right
                tw_o += tile_w_o                   
                filters = 0 
                out_channels = 0
                for k in range(command_array[0,i,j].get_tile_loads()): # load number
                
                    filters_start = filters
                    filters += command_array[k,i,j].get_filters()
                    load_weights = q_weights[...,filters_start:filters]
                    if use_bias:
                        load_bias = q_bias[...,filters_start:filters]    
                    
                    if command_array[k,i,j].tile_width == input_shape[1]:
                        command_array[k,i,j].set_scattered_lines(0)
                        
                    
                    if command_array[k,i,j].get_new_weights() == 0:
                        weight_address_iter = addr_cycle(weight_addr_init,0,self.cache_sizes["Weight cache block size"])
                    
                    collector_nbr = self.syst_array_struct['COLLECTOR_NBR'] 
                    syst_array_width = int(self.syst_array_struct['SYST_ARRAY_WIDTH'])
                    tile_recurrence = command_array[0,i,j].get_tile_recurrence()
                    syst_array_width *= tile_recurrence
                    collector_nbr *= tile_recurrence
                    weight_splited = []
                    bias_splited = []
                    outshift_splited = []
                    wr = load_weights.reshape(load_weights.shape[:-1]+
                                              (int(load_weights.shape[-1]/syst_array_width),
                                               syst_array_width))
                    if use_bias:
                        br = load_bias.reshape(int(load_bias.shape[-1]/syst_array_width),
                                               syst_array_width)
                    for split in range(collector_nbr):
                        weight_splited.append(wr[...,split::collector_nbr])
                        if use_bias:
                            bias_splited.append(br[...,split::collector_nbr])
                            
                    weights_resh = np.concatenate(weight_splited,axis=-1)
                    weights_resh = weights_resh.reshape(load_weights.shape)
                    if use_bias:
                        bias_resh = np.concatenate(bias_splited,axis=-1)
                        bias_resh = bias_resh.reshape(load_bias.shape)
                    
                    command_binaries = []
                    in_tile_info = []
                    weight_info = []
                    for l in range(in_channels):
                        if l == 0:
                            command_array[k,i,j].set_first_channel(1)
                        else: 
                            command_array[k,i,j].set_first_channel(0)  
                            command_array[k,i,j].set_reset_weight_cache(0) 
                        
                        if l == in_channels-1:
                            command_array[k,i,j].set_last_channel(1)
                        else: 
                            command_array[k,i,j].set_last_channel(0)                            
                        
                        vectorized_weights = self.float6_binary_to_32bit_vectors(weights_resh[:,:,l,:],
                                                                                 command_array[k,i,j].get_tile_recurrence())
                        if vectorized_weights.size != command_array[k,i,j].get_weight_length():
                            raise Exception("Weight length {} in command {} {} does not fit " \
                                            "to length of vecotized weights size of {}".format(command_array[k,i,j].get_weight_length(),i,j,
                                            vectorized_weights.size))
                        if use_bias and l==0:
                            if self.quantization['BIAS_WIDTH'] == 4:
                                vectorized_bias = self.fixed4_binary_to_32bit_vectors(bias_resh)
                            elif self.quantization['BIAS_WIDTH'] == 8:
                                vectorized_bias = self.fixed8_binary_to_32bit_vectors(bias_resh)
                            elif self.quantization['BIAS_WIDTH'] == 6:
                                vectorized_bias = self.fixed6_binary_to_32bit_vectors(bias_resh)
                            else:
                                raise NotImplementedError("Bias width: {} not supported yet.".format(self.quantizeation['BIAS_WIDTH']))                            
                            if vectorized_bias.size != command_array[k,i,j].get_filters()/(32/self.quantization['BIAS_WIDTH']):
                                raise Exception("Bias length of {}  in command {}{} does not fit" \
                                                "to vectorized bias size {}".format(
                                                    command_array[k,i,j].get_filters()/(32/self.quantization['BIAS_WIDTH']),i,j,
                                                    vectorized_bias.size))
                            command_array[k,i,j].set_use_bias(1)
                            vectorized_weights = np.append(vectorized_bias,vectorized_weights)
                        else:
                            command_array[k,i,j].set_use_bias(0)
                            
                        if l == 0 and k == 0 and i == 0 and j == 0:
                            if layer.activation == None or layer.activation == tf.keras.activations.linear:
                                layer_activation_sel = 0
                            elif layer.activation == tf.keras.activations.relu:
                                layer_activation_sel = 1
                            else:
                                raise NotImplementedError("Activation " + str(layer.activation) + " not supported") 
                            layer_settings_vec = vectorize_layer_settings(layer.out_shift,layer.bias_shift,layer_activation_sel,
                                                                          self.layer_settings_inpr,self.quantization['MAX_OUT_SHIFT'])
                            vectorized_weights = np.append(layer_settings_vec,vectorized_weights)
                            command_array[k,i,j].set_new_layer_settings(1)
                        else:
                            command_array[k,i,j].set_new_layer_settings(0)

                                
                        weight_addr = weight_address_iter(kernel_size[0]*kernel_size[1]*command_array[k,i,j].get_iterations())
                        command_array[k,i,j].set_weight_address(weight_addr)
                        command_dict, weight_dict = command_array[k,i,j].to_dict()
                        weight_id = [command_array[k,i,j].get_new_weights(),command_array[k,i,j].get_weight_address()]
                        tile_h = command_array[k,i,j].get_tile_height()
                        tile_w = command_array[k,i,j].get_tile_width()                       
                        self._Fmap_translator.set_commands(command_dict)
                        self._Weight_translator.set_commands(weight_dict) 
                        cm32 = self._Fmap_translator.write_commands_32()
                        wcm32 =self._Weight_translator.write_commands_32()
     
                        if command_array[k,i,j].get_new_weights() != 0:
                            vectorized_weights = np.insert(vectorized_weights,0,np.int32(wcm32),axis=0)
                        else: 
                            vectorized_weights = np.int32(wcm32)
                            
                        command_binaries.append(np.append(vectorized_weights,cm32))   
                        in_tile_info.append(np.array([layer_idx,i,j,k,l],dtype=np.uint32))
                        weight_info.append(weight_id)
    
                    weight_info_array[k,i,j] = weight_info
                    in_tile_info_array[k,i,j] = in_tile_info
                    command_binary_array[k,i,j] = command_binaries
                    in_tiles_array[k,i,j] = np.array([th,th+command_array[k,i,j].get_tile_height(),
                                        tw,tw+command_array[k,i,j].get_tile_width()],dtype=np.uint32)
                           
                    tile_h_o = command_array[k,i,j].get_output_tile_height()
                    tile_w_o = command_array[k,i,j].get_output_tile_width()  
                                          
                    out_tiles_array[k,i,j] = np.array([th_o,th_o+command_array[k,i,j].get_output_tile_height(),
                                     tw_o,tw_o+command_array[k,i,j].get_output_tile_width(),
                                     out_channels,
                                     command_array[k,i,j].get_iterations()*command_array[k,i,j].get_tile_recurrence()*
                                     self.syst_array_struct['SYST_ARRAY_WIDTH']+out_channels,
                                     command_array[k,i,j].get_scattered_lines()],dtype=np.uint32) 
                    
                    out_channels += command_array[k,i,j].get_iterations()*command_array[k,i,j].get_tile_recurrence()*self.syst_array_struct['SYST_ARRAY_WIDTH']
       

      
        weight_addr = weight_address_iter(kernel_size[0]*kernel_size[1]*command_array[-1,-1,-1].get_iterations())
        set_weight_addr_vec = np.vectorize(Command.set_weight_address,otypes=[object])        
        set_weight_addr_vec(command_array,weight_addr)          
        command_list = []
        command_binaries = []
        in_tiles = []
        out_tiles = []
        in_tile_info = []
        weight_info = []
        
        for i in range(command_array.shape[0]):
            for j in range(command_array.shape[1]):
                for k in range(command_array.shape[2]):
                    if i >= command_array[i,j,k].get_tile_loads():
                        continue
                    command_list.append(command_array[i,j,k])
                    command_binaries.extend(command_binary_array[i,j,k])
                    in_tiles.append(in_tiles_array[i,j,k])
                    out_tiles.append(out_tiles_array[i,j,k])
                    in_tile_info.extend(in_tile_info_array[i,j,k])
                    weight_info.extend(weight_info_array[i,j,k])                   
    
        return command_list, command_binaries, in_tiles, in_tile_info, out_tiles, weight_info, weight_addr
            
        
    def translate_inversebottleneck(self,depth_wise_layer : layers.DepthwiseConv2D ,conv2d_1x1_layer : layers.Conv2D):
        """
        Creates a command list for the Intuitus hw accelerator based on the 
        properties of a Keras conv2d 1x1 layer class followed by a depthwise
        separable conv2d layer, which form the expansion part of an inverse 
        bottleneck layer. This combination is treated differnt in order to 
        increase the local data reuse and therefore reduce the required 
        memory bandwidth and power consumption. 

        Parameters
        ----------
        depth_wise_layer : tensorflow.python.keras.layers.convolutional.DepthwiseConv2D
            Keras DepthwiseConv2D layer.
        conv2d_1x1_layer : tensorflow.python.keras.layers.convolutional.Conv2D
            Keras Conv2d layer with kernel_size = (1,1).

        Raises
        ------
        NotImplementedError
            Since Keras offers a higher flexibility than Intuitus the range 
            of the kernel_size and strides is restiricted.

        Returns
        -------
        command_list :  np.array [Th,Tw,Ci,Trld]
            Th... Tile number in height axis  
            Tw... Tile number in width axis 
            Ci... Input channel number 
            Trld... Tile reload number. (Depends on output channel number, Image shape and Intuitus hyperparameter)
            Array of commands for each image tile of a layer input. 

        """
        raise NotImplementedError("Inverse Bottleneck is not implemented yet.")
        
    def run_layer(self,layer_nbr : int, fmap):
        return self.keras_model.layers[layer_nbr].sim_hw(fmap)
    
    def get_model(self):
        return self.keras_model
    
    def set_model(self,model :models.Model):
        self.keras_model = model
    
    def save_model(self,filepath):
        models.save_model(self.keras_model,filepath)
        
    def load_model(self,filepath):
        self.keras_model = models.load_model(filepath)
        
    def get_layers(self):
        return self.keras_model.layers

    def get_weights(self):
        return self.keras_model.get_weights()
    def load_weights(self, filepath, by_name=False, skip_mismatch=False):
        return self.keras_model.load_weights(filepath,by_name,skip_mismatch)
    def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
        return self.keras_model.compile(optimizer, loss, metrics, loss_weights,sample_weight_mode,weighted_metrics,
                                        target_tensors,distribute,**kwargs)
    @property
    def metrics(self):
        return self.keras_model.metrics()
    
    @property
    def metrics_names(self):
        return self.keras_model.metrics_names()    
    
    @property
    def run_eagerly(self):
        return self.keras_model.run_eagerly() 
    
    @run_eagerly.setter
    def run_eagerly(self, value):
        self.keras_model.run_eagerly(value)
    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):    
        return self.keras_model.fit(x,y,batch_size,epochs,verbose,callbacks,validation_split,validation_data,shuffle,
                                    class_weight,sample_weight,initial_epoch,steps_per_epoch,validation_steps,validation_freq,
                                    max_queue_size,workers,use_multiprocessing,**kwargs)
    def evaluate(self,
                x=None,
                y=None,
                batch_size=None,
                verbose=1,
                sample_weight=None,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        return self.keras_model.evaluate(x,y,batch_size,verbose,sample_weight,steps,callbacks,max_queue_size,workers,
                                        use_multiprocessing)
   
    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        return self.keras_model.evaluate(x,batch_size,verbose,steps,callbacks,max_queue_size,workers,use_multiprocessing)
    
    def reset_metrics(self):
        self.keras_model.reset_metrics()

    def train_on_batch(self,
                        x,
                        y=None,
                        sample_weight=None,
                        class_weight=None,
                        reset_metrics=True):
        return self.keras_model.train_on_batch(x,y,sample_weight,class_weight,reset_metrics)
    
    def test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True):
        return self.keras_model.test_on_batch(x,y,sample_weight,reset_metrics)
    
    def predict_on_batch(self, x):  
        return self.keras_model.predict_on_batch(x)
    
    @property
    def sample_weights(self):
      return self.keras_model.sample_weight