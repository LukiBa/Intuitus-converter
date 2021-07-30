# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:20:41 2020

@author: Lukas Baischer
"""

import json
import pathlib
import numpy as np
import logging 
from typing import Tuple 

from intuitus_converter.core import float8, fixed4
        
                    
class Intuitus_Basis(): 
    def __init__(self,HyperPar_Path: pathlib.Path):
        logging.getLogger(__name__)
        with open(HyperPar_Path) as json_file:
            self.data = json.load(json_file)
            self.hyper_parameter_commands = self.data['Command Interpreter']
            self.command_list= []
            for i in sorted(self.hyper_parameter_commands['Sequence'].keys()):
                self.command_list.append(self.hyper_parameter_commands['Sequence'][i])  
            self.bit_widths = self.hyper_parameter_commands['Bit widths'] 
            self.description = self.hyper_parameter_commands['Description'] 
            self.possible_values = self.hyper_parameter_commands['Possible values']
            self.layer_settings_inpr = self.data['Layer settings interpreter']
            
            
            self.quantization = self.data['Quantization']
            self.zero_value = np.uint8(int(str(self.data['Zero value']),16))
            self.syst_array_struct = self.data['Systolic array structure'] 
            self.cache_sizes = self.data['Cache sizes']
            self.pu_mode = self.data['PU Mode']
        
        self.translate_exec = 0
        self.max_tile_width = 2**self.bit_widths['tile_width'] # no -1 because range is 1 to 2**bitwidth (tile width = 0 would not be usefull anyway)
        self.HyperPar_Path = HyperPar_Path
        logging.info("---------------------------------------------------------------")
        logging.info("Use: " + self.data['Name'] + " Version: " +  str(self.data['Version']))
        logging.info("---------------------------------------------------------------")   
        
    def print_intuitus_version(self):
        print("---------------------------------------------------------------")
        print("Use: " + self.data['Name'] + " Version: " +  str(self.data['Version']))
        print("---------------------------------------------------------------")   
        
    def quantize_img(self,img):
        return float8(img)
 
    def quantize_weights_and_bias(self,weights_and_bias):
        if not isinstance(weights_and_bias,list):
            raise AttributeError("No list used for weights and bias")
        q_wb = []
        for i in weights_and_bias:
            if len(i.shape) == 1:
                q_wb.append(self.quantize_bias(i))
            else:
                q_wb.append(self.quantize_weights(i))
        return q_wb
    def quantize_weights(self,weights):
        mantissa_width = self.quantization['WEIGHT_MANTISSA_WIDTH']
        exp_width = self.quantization['WEIGHT_EXPONENT_WIDTH']
        #weights = weights/(np.max(np.abs(weights)))
        
        if isinstance(weights,list):
            q_weights = []
            for weight_np in weights:
                
                sign = np.uint8(np.where(weight_np<0.0,1,0))
                value = np.abs(weight_np)
                exp = np.where(value==0.0,(2**exp_width)-1,(-1)*np.log2(value))
                exp = np.uint8(np.clip(exp,0,2**(exp_width)-1))                
                mantissa = np.uint8(np.around(value*2**(np.int32(exp)+mantissa_width)))
                mantissa = np.uint8(np.clip(mantissa,0,(2**mantissa_width)-1)) 
                q_value = np.float16(np.where(sign==0,mantissa*2.0**((-1)*exp-mantissa_width),
                                              (-1)*mantissa*2.0**((-1)*exp-mantissa_width)))        
                q_weights.append(q_value)
            return q_weights
                       
        else:
            sign = np.uint8(np.where(weights<0.0,1,0))
            value = np.abs(weights)
            exp = np.where(value==0.0,(2**exp_width)-1,(-1)*np.log2(value))
            exp = np.uint8(np.clip(exp,0,2**(exp_width)-1))                
            mantissa = np.uint8(np.around(value*2**(np.int32(exp)+mantissa_width)))
            mantissa = np.uint8(np.clip(mantissa,0,(2**mantissa_width)-1)) 
            q_value = np.float16(np.where(sign==0,mantissa*2.0**((-1)*exp-mantissa_width),
                                          (-1)*mantissa*2.0**((-1)*exp-mantissa_width)))       
            return q_value  

    def quantize_bias(self,bias):
        fixed_bias = self.bias_to_fixed4(bias)
        return fixed_bias.to_float32()

    def bias_to_fixed4(self,bias):
        bias_width = self.quantization['BIAS_WIDTH']
        exponent = self.quantization['BIAS_EXPONENT']
        
        # bias_mantissa = np.int8(np.around(2**(exponent+bias_width-1)*bias))
        # bias_mantissa = np.clip(bias_mantissa,(-1)*2**(bias_width-1),2**(bias_width-1)-1)
        return fixed4(bias,exponent)
    
    def fixed4_binary_to_32bit_vectors(self,values : np.ndarray):
        serialized = np.bitwise_and(15,np.int32(values.reshape(values.size)))
        vectors = np.zeros([int(np.ceil(values.size/8))],dtype=np.int32)
        for i in range(vectors.size):
            for j in range(8):
                vectors[i] = np.bitwise_or(vectors[i],np.left_shift(serialized[i*8+j],4*j)) 
        return vectors
    
    def fixed8_binary_to_32bit_vectors(self,values : np.ndarray):
        serialized = np.bitwise_and(255,np.int32(values.reshape(values.size)))
        vectors = np.zeros([int(np.ceil(values.size/4))],dtype=np.int32)
        for i in range(vectors.size):
            for j in range(4):
                vectors[i] = np.bitwise_or(vectors[i],np.left_shift(serialized[i*4+j],8*j)) 
        return vectors   

    def fixed6_binary_to_32bit_vectors(self,values : np.ndarray):    
        if values.shape[-1]%16 != 0:
            raise NotImplementedError("Filter number must be a multiple of {} due to hardware restrictions. Fill up with 0 Filters in order to use current design.".format(16))        
        vectors = np.zeros([int(values.shape[0]/16),3],dtype=np.int32)
        for k in range(vectors.shape[0]):
            serialized = np.int32(values[k*16:(k+1)*16])
            serialized = np.bitwise_and(serialized,63)
            vectors[k,0] = serialized[0]
            for j in range(1,6,1):
                vectors[k,0] = np.bitwise_or(vectors[k,0],np.left_shift(serialized[j],j*6))
            
            vectors[k,1] = np.right_shift(serialized[5],2)
            for j in range(1,6,1):
                vectors[k,1] = np.bitwise_or(vectors[k,1],np.left_shift(serialized[j+5],j*6-2))  
                
            vectors[k,2] = np.right_shift(serialized[10],4)
            for j in range(1,6,1):
                vectors[k,2] = np.bitwise_or(vectors[k,2],np.left_shift(serialized[j+10],j*6-4))  
                
        return vectors.reshape(vectors.size) 
    
    def shfits_2bit_to_32bit_vectors(self,values : np.ndarray):
        serialized = np.bitwise_and(3,np.int32(values.reshape(values.size)))
        vectors = np.zeros([int(np.ceil(values.size/16))],dtype=np.int32)
        for i in range(vectors.size):
            for j in range(16):
                vectors[i] = np.bitwise_or(vectors[i],np.left_shift(serialized[i*16+j],2*j)) 
        return vectors  
        
    def float8_to_32bit_vectors(self,values : np.ndarray,scattered_lines=1):
        """
        

        Parameters
        ----------
        values : np.ndarray[H,W,C]
            fmap values.

        Returns
        -------
        vectors : np.ndarray[H,W/4,C].
            vectorized 32bit values 

        """
        if not np.issubdtype(values.dtype, np.integer):
            raise AttributeError("Input array must be of type integer")
            
        if scattered_lines == 0:
            values = np.int32(values)
            vectors_shape = list(values.shape)
            if len(vectors_shape) == 3:
                values = np.moveaxis(values,-1,0)
                values = values.reshape(values.shape[0],values.shape[1]*values.shape[2])
                vectors = np.zeros((values.shape[0],int(np.ceil(values.shape[1]/4.0))), dtype=np.int32)
                for i in range(values.shape[1]):
                    vectors[:,int(i/4)] = np.bitwise_and(vectors[:,int(i/4)],np.left_shift(values[:,i],(i%4)*8))
            elif len(vectors_shape) == 2:
                values = values.reshape(values.shape[0]*values.shape[1])
                vectors = np.zeros((int(np.ceil(values.shape[0]/4.0))), dtype=np.int32)
                for i in range(values.shape[0]):
                    vectors[int(i/4)] = np.bitwise_or(vectors[int(i/4)],np.left_shift(values[i],(i%4)*8))  
            else:
                raise NotImplementedError("shpae with size {} not implemented".format(values.shape))
            
        else:    
            values = np.int32(values)
            vectors_shape = list(values.shape)
            vectors_shape[1] = int(np.ceil(vectors_shape[1] / 4))
            vectors = np.zeros(vectors_shape, dtype=np.int32)
            if len(vectors_shape) == 3:
                for i in range(0,values.shape[1]-4,4):
                    idx = int(i/4)
                    vectors[:,int(i/4),:] = values[:,i,:]
                    vectors[:,int(i/4),:] = np.bitwise_or(vectors[:,int(i/4),:],np.left_shift(values[:,i+1,:],8))   
                    vectors[:,int(i/4),:] = np.bitwise_or(vectors[:,int(i/4),:],np.left_shift(values[:,i+2,:],16))   
                    vectors[:,int(i/4),:] = np.bitwise_or(vectors[:,int(i/4),:],np.left_shift(values[:,i+3,:],24)) 
                    
                vectors[:,-1,:] = 0
                idx = 0
                for i in range(vectors.shape[1]*4-4,values.shape[1]):
                    vectors[:,-1,:] = np.bitwise_or(vectors[:,-1,:],np.left_shift(values[:,i,:],idx*8))
                    idx += 1
            elif len(vectors_shape) == 2:
                for i in range(0,values.shape[1]-4,4):
                    idx = int(i/4)
                    vectors[:,idx] = values[:,i]
                    vectors[:,idx] = np.bitwise_or(vectors[:,idx],np.left_shift(values[:,i+1],8))   
                    vectors[:,idx] = np.bitwise_or(vectors[:,idx],np.left_shift(values[:,i+2],16))   
                    vectors[:,idx] = np.bitwise_or(vectors[:,idx],np.left_shift(values[:,i+3],24))               
                
                vectors[:,-1] = 0
                idx = 0
                for i in range(vectors.shape[1]*4-4,values.shape[1]):
                    vectors[:,-1] = np.bitwise_or(vectors[:,-1],np.left_shift(values[:,i],idx*8))
                    idx += 1
            else:
                raise NotImplementedError("shpae with size {} not implemented".format(values.shape))                    
        return vectors 
                
    def float6_binary_to_32bit_vectors(self,values : np.ndarray, tile_recurrence):
        ram_nbr = self.syst_array_struct['WEIGHT_BRAM_NBR']
        if ram_nbr == 3:
            vectors=self._use_3_weight_ram(values)
        elif ram_nbr == 2:
            vectors=self._use_2_weight_ram(values)    
        else:
            raise NotImplementedError("Only implemented for weight bram number equals 2 or 3.")   
        
        if tile_recurrence == 2:
            out_vectors = np.zeros([int(vectors.shape[0]/2),vectors.shape[1]*2,vectors.shape[2]],dtype=np.int32)
            for i in range(out_vectors.shape[0]):
                for j in range(vectors.shape[1]):
                    out_vectors[i,j*2,:] = vectors[i*2,j,:]
                    out_vectors[i,j*2+1,:] = vectors[i*2+1,j,:]
        elif tile_recurrence == 4:
            out_vectors = np.zeros([int(vectors.shape[0]/4),vectors.shape[1]*4,vectors.shape[2]],dtype=np.int32)
            for i in range(out_vectors.shape[0]):
                for j in range(vectors.shape[1]):
                    out_vectors[i,j*4,:] = vectors[i*4,j,:]
                    out_vectors[i,j*4+1,:] = vectors[i*4+1,j,:]
                    out_vectors[i,j*4+2,:] = vectors[i*4+2,j,:]
                    out_vectors[i,j*4+3,:] = vectors[i*4+3,j,:]
        else:
            out_vectors = vectors
                    
        return out_vectors.reshape(vectors.size) 
            
    def _use_3_weight_ram(self,values : np.ndarray):
        if values.shape[-1]%16 != 0:
            raise NotImplementedError("Filter number must be a multiple of {} due to hardware restrictions. Fill up with 0 Filters in order to use current design.".format(16))        
        vectors = np.zeros([int(values.shape[-1]/16),values.shape[0]*values.shape[1],3],dtype=np.int32)
        for k in range(vectors.shape[0]):
            serialized = np.int32(values[:,:,k*16:(k+1)*16].reshape((vectors.shape[1],16)))
            serialized = np.bitwise_and(serialized,63)
            
            for i in range(vectors.shape[1]):
                vectors[k,i,0] = serialized[i,0]
                for j in range(1,6,1):
                    vectors[k,i,0] = np.bitwise_or(vectors[k,i,0],np.left_shift(serialized[i,j],j*6))
                
                vectors[k,i,1] = np.right_shift(serialized[i,5],2)
                for j in range(1,6,1):
                    vectors[k,i,1] = np.bitwise_or(vectors[k,i,1],np.left_shift(serialized[i,j+5],j*6-2))  
                    
                vectors[k,i,2] = np.right_shift(serialized[i,10],4)
                for j in range(1,6,1):
                    vectors[k,i,2] = np.bitwise_or(vectors[k,i,2],np.left_shift(serialized[i,j+10],j*6-4))  
        return vectors
                    
    def _use_2_weight_ram(self,values : np.ndarray):
       if values.shape[-1]%8 != 0:
           raise NotImplementedError("Filter number must be a multiple of {} due to hardware restrictions. Fill up with 0 Filters in order to use current design.".format(8))             
       vectors = np.zeros([int(values.shape[-1]/8),values.shape[0]*values.shape[1],2],dtype=np.int32)
       for k in range(vectors.shape[0]):
           serialized = np.int32(values[:,:,k*8:(k+1)*8].reshape((vectors.shape[1],8)))
           
           for i in range(vectors.shape[1]):
               vectors[k,i,0] = serialized[i,0]
               for j in range(1,6,1):
                   vectors[k,i,0] = np.bitwise_or(vectors[k,i,0],np.left_shift(serialized[i,j],j*6))
               
               vectors[k,i,1] = np.right_shift(serialized[i,5],2)
               for j in range(1,3,1):
                   vectors[k,i,1] = np.bitwise_or(vectors[k,i,1],np.left_shift(serialized[i,j+5],j*6-2))  
                   
       return vectors                  
                    
            
    def _get_tile_height(self,input_height : int,kernel_size : int,stride : int,max_pooling : int):
        syst_array_line_nbr = self.syst_array_struct['SYST_ARRAY_HEIGHT']*self.syst_array_struct['PE_HEIGHT']
        max_tile_height = syst_array_line_nbr*stride
        if input_height <= max_tile_height: 
            return [input_height],[int(input_height/stride)]
        else:
            overlays_btm = np.int32(np.floor(kernel_size/2)) 
            if stride == 1: 
                overlays_top = overlays_btm
            else:
                overlays_top = overlays_btm-1
                if input_height%2 ==1: # For odd image height
                    raise NotImplementedError("Using stride 2 for odd image size is not supported by hardware.")
                    overlays_btm -= 1

            if max_pooling == 1:
                stride = 2                  
                
            tile_height_in = []
            tile_height_out = []
            for i in range(int(np.floor(input_height/max_tile_height))):
                if i == int(np.floor(input_height/max_tile_height))-1:
                    tile_height_in.append(max_tile_height+overlays_top)
                elif i == 0:
                    tile_height_in.append(max_tile_height+overlays_btm)
                else:
                    tile_height_in.append(int(max_tile_height+overlays_btm+overlays_top))
                    
                tile_height_out.append(int(max_tile_height/stride))
                
            if input_height%max_tile_height != 0:
                if int(np.floor(input_height/max_tile_height)) > 1:
                        tile_height_in[-1] += overlays_btm
                        
                if input_height%max_tile_height < 4:
                    diff = 4 - (input_height%max_tile_height)
                    height = 4
                    tile_height_in[-1] -= diff
                    if stride == 2 and (tile_height_in[-1] %2 != 0 or diff %2 != 0):
                        raise Exception("Stride 2 requires even images size")
                    tile_height_out[-1] -= int(diff/stride)
                    tile_height_in.append(height+overlays_top)
                    tile_height_out.append(int(height/stride))                       
                else:     
                    tile_height_in.append(input_height%max_tile_height+overlays_top)
                    tile_height_out.append(int((input_height%max_tile_height)/stride))
                
            
            return tile_height_in, tile_height_out

    def _get_tile_width(self,input_width : int,kernel_size : int, stride: int, max_pooling : int, dsp_feedback_len : int = 3):
        if stride == 2 and input_width%2 ==1:  # For odd image width
                raise NotImplementedError("Using stride 2 for odd image size is not supported by hardware.")
                
        if input_width <= self.max_tile_width:
            if max_pooling == 1:
                stride = 2              
            return [input_width], [int(input_width/stride)]
            
        # search for optimal tile width 
        overlays_right = np.int32(np.floor(kernel_size/2)) 
        if stride == 1: 
            overlays_left = overlays_right
        else:
            overlays_left = overlays_right-1
            if input_width%2 ==1:  # For odd image width
                overlays_right -= 1
                
        if max_pooling == 1:
            stride = 2            
        
        # right most tile 
        x = int(self.max_tile_width) - overlays_left
        while(x % (4*stride) != 0):
            x -=1
        
        tile_width_in = [x+overlays_left]
        tile_width_out = [x]
        rest_width = input_width - tile_width_out[-1]
        
        x = int(self.max_tile_width) - overlays_left -overlays_right
        while(x % (4*stride) != 0):
            x -=1    

        while(rest_width > x):
            tile_width_in.insert(0,int(x + overlays_left + overlays_right))
            tile_width_out.insert(0,x)        
            rest_width = rest_width - tile_width_out[-1]
            
        tile_width_in.insert(0,rest_width + overlays_left)
        tile_width_out.insert(0,rest_width)           

        tile_width_out = [int(i/stride) for i in tile_width_out]

        return tile_width_in, tile_width_out
    
    def _get_tile_loads(self,filters : int, tile_size : Tuple[list,list],kernel_size,use_bias):
        if filters%self.syst_array_struct['SYST_ARRAY_WIDTH'] != 0:
            raise NotImplementedError("Filter numbers, which are no multiple of 16 are not implemented. Workaround: Extend your filters with zero filters")
        
        max_weights = self.cache_sizes['Weight cache block size']/2
        weight_nbr = kernel_size[0]*kernel_size[1]         
        weight_vecs_each_bram_row = self.syst_array_struct['WEIGHT_BRAM_NBR']
        
        iterations = np.ndarray([len(tile_size[0]),len(tile_size[1])],object)
        output_channels = np.ndarray([len(tile_size[0]),len(tile_size[1])],object)
        weight_widths = np.ndarray([len(tile_size[0]),len(tile_size[1])],object)
        weight_lengths = np.ndarray([len(tile_size[0]),len(tile_size[1])],object)
        tile_recurrences = np.zeros([len(tile_size[0]),len(tile_size[1])],dtype=np.int32)
        tile_loads = np.zeros([len(tile_size[0]),len(tile_size[1])],dtype=np.int32)
        for i in range(len(tile_size[0])):
            for j in range(len(tile_size[1])):
                #byte_nbr_each_pe_each_tile = self.syst_array_struct['PE_HEIGHT'] * tile_size[1][j] 
                tiles_each_pe = np.floor((self.cache_sizes['PE cache size']-self.cache_sizes['PE cach full offset']) 
                                         / (tile_size[1][j]))
                syst_array_line_nbr = self.syst_array_struct['SYST_ARRAY_HEIGHT']*self.syst_array_struct['PE_HEIGHT']
                if tile_size[0][i] <= syst_array_line_nbr/4 and filters >= 4*self.syst_array_struct['SYST_ARRAY_WIDTH']:
                    max_channels_each_iterations = self.syst_array_struct['SYST_ARRAY_WIDTH'] * 4
                    tile_recurrence = 4
                elif tile_size[0][i] <= syst_array_line_nbr/2 and filters >= 2*self.syst_array_struct['SYST_ARRAY_WIDTH']:
                    max_channels_each_iterations = self.syst_array_struct['SYST_ARRAY_WIDTH'] * 2
                    tile_recurrence = 2
                else:
                    max_channels_each_iterations = self.syst_array_struct['SYST_ARRAY_WIDTH']
                    
                    tile_recurrence = 1
                
                max_weights *= tile_recurrence
                tiles_each_pe = min(tiles_each_pe,filters/max_channels_each_iterations)
                tiles_each_pe = min(tiles_each_pe,2**self.bit_widths['iterations'])
                max_out_channels_each_load = tiles_each_pe * max_channels_each_iterations
                if min(filters,max_out_channels_each_load)*weight_nbr/self.syst_array_struct['SYST_ARRAY_WIDTH'] > max_weights: 
                        max_out_channels_each_load = np.floor(max_weights/weight_nbr)*self.syst_array_struct['SYST_ARRAY_WIDTH']
                        tiles_each_pe = int(max_out_channels_each_load/max_channels_each_iterations)
                load_nbr = np.int32(np.ceil(filters/max_out_channels_each_load))

                if load_nbr == 1:
                    iteration = np.array([filters/max_channels_each_iterations])
                else:
                    tiles_each_pe = np.floor((self.cache_sizes['PE cache size']/2) / (tile_size[1][j])) + 3 # for multiple loads only have of the cache is used eacht load to achieve better parallelizm
                    tiles_each_pe = min(tiles_each_pe,filters/max_channels_each_iterations)
                    tiles_each_pe = min(tiles_each_pe,2**self.bit_widths['iterations'])  
                    max_out_channels_each_load = tiles_each_pe * max_channels_each_iterations
                    if min(filters,max_out_channels_each_load)*weight_nbr/self.syst_array_struct['SYST_ARRAY_WIDTH'] > max_weights: 
                            max_out_channels_each_load = np.floor(max_weights/weight_nbr)*self.syst_array_struct['SYST_ARRAY_WIDTH']
                            tiles_each_pe = int(max_out_channels_each_load/max_channels_each_iterations)
                    load_nbr = np.int32(np.ceil(filters/max_out_channels_each_load))                    
                    iteration = np.zeros([load_nbr],dtype=np.int32)
                    iteration[:-1] = np.int32(tiles_each_pe)
                    if filters%tiles_each_pe == 0:
                        iteration[load_nbr-1] = tiles_each_pe
                    else:
                        iteration[load_nbr-1] = np.int32(int(filters/max_channels_each_iterations)%tiles_each_pe)
                    
                out_channels = np.zeros([load_nbr],dtype=np.int32)
                out_channels[:-1] = max_channels_each_iterations*iteration[:-1]
                if filters%max_channels_each_iterations == 0:
                    out_channels[load_nbr-1] = max_channels_each_iterations*iteration[load_nbr-1]
                else:
                    out_channels[load_nbr-1] = (filters%max_channels_each_iterations)*iteration[load_nbr-1]
                    
                weight_width = np.zeros([load_nbr],dtype=np.int32)
                weight_width[:-1] =  np.int32(self.syst_array_struct['SYST_ARRAY_WIDTH'])
                weight_width[load_nbr-1] = np.int32(np.ceil(out_channels[-1]/iteration[-1]))
                
                weight_length = np.zeros([load_nbr],dtype=np.int32)
                weight_length = np.int32(weight_nbr*weight_vecs_each_bram_row*iteration*tile_recurrence)
                    
                iterations[i,j] = iteration
                tile_recurrences[i,j] = tile_recurrence
                tile_loads[i,j] = np.int32(load_nbr)
                output_channels[i,j] = out_channels
                weight_widths[i,j] = weight_width
                weight_lengths[i,j] = weight_length
                
        return iterations, output_channels, tile_recurrences, tile_loads, weight_lengths, weight_widths
    def reshape_tile_for_stride_2(self,tile):
        """
        Reshapes tile array 

        Parameters
        ----------
        tile : np.array [H,W,C] or [H,W]
            image tile.

        Returns
        -------
        tile_reshape : np.array [H,W,C] or [H,W].

        """
        if len(tile.shape) == 3: 
            tile_reshape = tile[0:-1:2,:,:]
            np.append(tile_reshape,tile[1:-1:2,:,:])
        else:
            tile_reshape = tile[0:-1:2,:]
            tile_reshape = np.concatenate((tile_reshape,tile[1::2,:]))         

        return tile_reshape

    def tile_img(self,img,kernel_size:int,stride:int): 
        """
        Tiles an image or feature map for the use in intuitus 

        Parameters
        ----------
        img : np.array [H,W,C]
            image or feature map.
        kernel_size : integer 
            Integer specifying the height and width of the 2d convolution window.
            Used for determining the overlaying collumns 
        stride : integer 
            Integer specifiying stride of 2d convolution

        Returns
        -------
        tiles : np.array [Th,Tw,C,H,W]
            Image tile for computation in intuitus.

        """
        input_shape = img.shape
        tile_heights = self._get_tile_height(input_shape[0],kernel_size,stride)
        tile_widths = self._get_tile_width(input_shape[1],kernel_size,stride)
        tile_sizes = (tile_heights[0], tile_widths[0])
        tile_number = (len(tile_sizes[0]), len(tile_sizes[1]))
        overlays_right = np.int32(np.floor(kernel_size/2))
        if stride == 1:
            overlays_left = overlays_right
        else:
            overlays_left = overlays_right-1          

        tiles = []
        tile_h = overlays_left +overlays_right
        th = 0
        for i in range(tile_number[0]):
            th += tile_h -overlays_left - overlays_right
            tw = 0
            tile_w = overlays_left +overlays_right
            for j in range(tile_number[1]):
                tw += tile_w - overlays_left - overlays_right
                tiles.append(img[th:th+tile_sizes[0][i],tw:tw+tile_sizes[1][j]])
                tile_h = tile_sizes[0][i]
                tile_w = tile_sizes[1][j]
        return tiles , tile_number         
        
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