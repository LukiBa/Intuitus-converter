# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:21:14 2020

@author: lukas
"""
import json
import pathlib
import numpy as np



class Command(): 
    def __init__(self, op_mode, stride, use_bias, tile_height, tile_width, padding, iterations, 
                 tile_recurrence, tile_loads, output_tile_height, output_tile_width, filters,
                 in_channels,weight_length,weight_width,max_pooling,scattered_lines,bias_width,
                 new_layer_settings):
        self.op_mode = op_mode
        self.stride = stride
        self.use_bias = use_bias
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.padding = padding
        self.iterations = np.int32(iterations)
        self.tile_recurrence = tile_recurrence
        self.tile_loads =  tile_loads
        self.output_tile_height = output_tile_height
        self.output_tile_width = output_tile_width        
        self.filters = filters
        self.in_channels = in_channels
        self.weight_width = weight_width
        self.weight_length = weight_length
        self.weight_address = 0 
        self.new_weights = 0
        self.sum_to_previous = 0
        self.last_channel = 0
        self.max_pooling = max_pooling
        self.reset_weight_cache = 0
        self.scattered_lines = scattered_lines
        self.bias_width = bias_width 
        self.new_layer_settings= new_layer_settings

    def set_op_mode(self,op_mode):
        self.op_mode = op_mode
    def set_stride(self,stride):
        self.stride = stride
    def set_use_bias(self,use_bias):
        self.use_bias = use_bias   
    def set_tile_height(self,tile_height):
        self.tile_height = tile_height       
    def set_tile_width(self,tile_width):
        self.tile_width = tile_width
    def set_padding(self,padding):
        self.padding = padding
    def add_padding(self,padding):
        self.padding = np.bitwise_or(self.padding,padding)
    def set_iterations(self,iterations):
        self.iterations = iterations
    def set_tile_recurrence(self,tile_recurrence):
        self.tile_recurrence = tile_recurrence  
    def set_tile_loads(self,tile_loads):
        self.tile_loads = tile_loads 
    def set_output_tile_height(self,output_tile_height):
        self.output_tile_height = output_tile_height       
    def set_output_tile_width(self,output_tile_width):
        self.output_tile_width = output_tile_width        
    def set_filters(self,filters):
        self.filters = filters
    def set_in_channels(self,in_channels):
        self.in_channels = in_channels
    def set_weight_width(self,weight_width):
        self.weight_width = weight_width
    def set_weight_length(self,weight_length):
        self.weight_length = weight_length
    def set_weight_address(self,weight_address):
        self.weight_address = weight_address
    def set_new_weights(self,new_weights):
        self.new_weights = new_weights        
    def set_first_channel(self,first_channel):
        self.first_channel = first_channel        
    def set_last_channel(self,last_channel):
        self.last_channel = last_channel           
    def set_max_pooling(self,max_pooling):
        self.max_pooling = max_pooling             
    def set_reset_weight_cache(self,reset_weight_cache):
       self.reset_weight_cache = reset_weight_cache  
    def set_scattered_lines(self,scattered_lines):
       self.scattered_lines = scattered_lines 
    def set_bit_width(self,bit_width):
       self.bit_width = bit_width          
    def set_new_layer_settings(self,new_layer_settings):
       self.new_layer_settings= new_layer_settings          
        
    def get_op_mode(self):
        return self.op_mode
    def get_stride(self):
        return self.stride
    def get_use_bias(self):
        return self.use_bias    
    def get_tile_height(self):
        return self.tile_height    
    def get_tile_width(self):
        return self.tile_width
    def get_padding(self):
        return self.padding
    def get_iterations(self):
        return self.iterations
    def get_tile_recurrence(self):
        return self.tile_recurrence
    def get_tile_loads(self):
       return self.tile_loads
    def get_output_tile_height(self):
        return self.output_tile_height    
    def get_output_tile_width(self):
        return self.output_tile_width   
    def get_filters(self):
       return self.filters
    def get_in_channels(self):
       return self.in_channels   
    def get_weight_width(self):
       return self.weight_width   
    def get_weight_length(self):
       return self.weight_length   
    def get_weight_address(self):
       return self.weight_address  
    def get_new_weights(self):
       return self.new_weights     
    def get_first_channel(self):
       return self.first_channel     
    def get_last_channel(self):
       return self.last_channel    
    def get_max_pooling(self):
       return self.max_pooling      
    def get_reset_weight_cache(self):
       return self.reset_weight_cache
    def get_scattered_lines(self):
       return self.scattered_lines    
    def get_bit_width(self):
       return self.bit_width  
    def get_new_layer_settings(self):
       return self.new_layer_settings 
       
    def to_dict(self):
        command_dict = {}
        command_dict['op_mode'] = self.op_mode
        command_dict['stride'] = self.stride
        command_dict["tile_height"] = self.tile_height
        command_dict["tile_width"] = self.tile_width-1
        command_dict["first_channel"] = self.first_channel
        command_dict["last_channel"] = self.last_channel
        command_dict["padding"] = self.padding
        command_dict["iterations"] = self.iterations-1
        command_dict["tile_recurrence"] = self.tile_recurrence-1
        # if self.first_channel != 0:
        #     command_dict['use_bias'] = int(self.use_bias)
        # else:
        #     command_dict['use_bias'] = 0
        command_dict["max_pooling"] = self.max_pooling
        command_dict["scattered_lines"] = self.scattered_lines
            
        weight_dict = {}
        weight_dict['weight_address'] = self.weight_address
        weight_dict['reset_weight_cache'] = self.reset_weight_cache
        weight_dict['new_weights'] = self.new_weights
        weight_dict['weight_length'] = self.weight_length/3
        weight_dict['weight_width'] = self.weight_width
        weight_dict['tile_recurrence'] = self.tile_recurrence-1
        weight_dict['new_layer_settings'] = self.new_layer_settings
        if self.use_bias:    
            weight_dict['bias_length'] = self.filters/(32/self.bias_width)/3
        else:
            weight_dict['bias_length'] = 0

        return command_dict, weight_dict
    
        

class Fmap_Command_interpreter(): 
    def __init__(self,HyperPar_Path: pathlib.Path):
        with open(HyperPar_Path) as json_file:
            self.data = json.load(json_file)
            
            self.hyper_parameter = self.data['Command Interpreter']
            self.command_list= []
            for i in sorted(self.hyper_parameter['Sequence'].keys()):
                self.command_list.append(self.hyper_parameter['Sequence'][i])
                
            self.bit_widths = self.hyper_parameter['Bit widths'] 
            self.description = self.hyper_parameter['Description'] 
            self.possible_values = self.hyper_parameter['Possible values']
            self.command_dict = {}
            
    def print_intuitus_version(self):
        print("---------------------------------------------------------------")
        print("Use: " + self.data['Name'] + " Version: " +  str(self.data['Version']))
        print("---------------------------------------------------------------")            
    def get_command_list(self): 
        """
        Command list getter 

        Returns
        -------
        List of String
            Sorted list of possible commands.

        """
        return self.command_list
    def get_command_dict(self): 
        """
        Command dict getter 

        Returns
        -------
        Dictionary 
            Sorted dictionary of settetd commands

        """
        return self.command_dict    
    def get_help(self,command=None): 
        """
        Provides a Description of a specific command or for all commands if 
        command parameter is set to None. 

        Parameters
        ----------
        command : String, optional
            Command string. The default is None.

        Returns
        -------
        String
            Description of the command or dictionary including a description of
            all possible commands.

        """
        if command == None:
            return self.description
        else:
            return self.description[command]
        
    def get_possible_value(self,command=None):
        """
        Returns possible values as a dictionary or as a String. 

        Parameters
        ----------
        command : String, optional
            Command string. The default is None.

        Returns
        -------
        String or Dictionary 

        """
        if command == None:
            return self.possible_values
        else:
            return self.possible_values[command]
    def set_commands(self,command_list : dict):
        """
        Command setter. 

        Parameters
        ----------
        command_list : dict 
            Dict of not necessarily all commands 

        Raises
        ------
        AttributeError
            If the type of the attribute is not dict.

        Returns
        -------
        None.

        """
        if isinstance(command_list,dict):
            for i in command_list.keys():
                if i == "op_mode":
                    self.command_dict[i] = self.possible_values[i][command_list[i]]
                elif i == "stride":
                    if command_list[i] == 1:
                        self.command_dict[i] = 0
                    elif command_list[i] == 2:
                        self.command_dict[i] = 1
                    else:
                        raise NotImplementedError("Only stride 1 or stride 2 is supported yet.")
                else:
                    self.command_dict[i] = command_list[i]
        else:
            raise AttributeError
        
            
    def write_commands_32(self):
        """
        Creates a 32 bit command vector

        Returns
        -------
        command32 : np.int32
            32 command vector.

        """
        command32 = np.uint32(0)
        shift_width = 0
        for i in self.command_list:
            mask = 2**self.bit_widths[i] -1
            masked_command = np.bitwise_and(np.uint32(self.command_dict[i]), mask)
            shifted_command = np.left_shift(masked_command,shift_width)
            command32 = np.bitwise_or(shifted_command,command32) 
            shift_width += self.bit_widths[i]
        return np.int32(command32)
        
class Weight_Command_interprter(Fmap_Command_interpreter):
    def __init__(self,HyperPar_Path: pathlib.Path):
        with open(HyperPar_Path) as json_file:
            self.data = json.load(json_file)
            
            self.hyper_parameter = self.data['Weight command interpreter']
            self.command_list= []
            for i in sorted(self.hyper_parameter['Sequence'].keys()):
                self.command_list.append(self.hyper_parameter['Sequence'][i])
                
            self.bit_widths = self.hyper_parameter['Bit widths'] 
            self.description = self.hyper_parameter['Description'] 
            self.possible_values = self.hyper_parameter['Possible values']
            self.command_dict = {}
            
def vectorize_layer_settings(out_shift, bias_shift, activation_select, layer_set_interpr_dict, max_out_shfit):
    out_shift_vec = np.int32(np.clip(out_shift,0,max_out_shfit))
    if out_shift_vec == 4:
        out_shift_vec = np.int32(7) #MSB indicates X left shift by 1. 1 downto 0 indicates postshift --> 4: 1 x shift 3 postshift --> 7
        bias_shift = np.int32(bias_shift-1)
    else:
        bias_shift = np.int32(bias_shift)
        
    shift = layer_set_interpr_dict["Bit widths"]["out_shift"]    
    bias_shift_vec = np.left_shift(bias_shift,shift)
    out_shift_vec = np.bitwise_or(out_shift_vec,bias_shift_vec)   
    activation_sel_vec = np.bitwise_and(2**layer_set_interpr_dict["Bit widths"]["activation_sel"]-1,activation_select)
    shift += layer_set_interpr_dict["Bit widths"]["bias_shift"]   
    activation_sel_vec = np.left_shift(activation_sel_vec, shift)
    out_shift_vec = np.bitwise_or(out_shift_vec,activation_sel_vec)  
    return out_shift_vec
    
            