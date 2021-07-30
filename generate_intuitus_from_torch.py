# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:40:32 2021

@author: lukas
"""

#%% import public modules
import argparse
import pathlib
import timeit
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras_yolo.yolov4 import YOLO, decode
from keras_yolo.utils import load_config
from keras_yolo.backbone import darknet53_tiny_folding_bn,cspdarknet53_tiny_folding_bn


#%% import custom modules 
from intuitus_converter import Intuitus_Converter
from intuitus_converter.core import convert_torch2qKeras_param
#%% Parameter 
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='parameters/int8_6', help='path to folder containing parameter')
parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
parser.add_argument('--input_size', type=int, default=416, help='define input size of export model')
parser.add_argument('--model', type=str, default='yolov3', help='yolov3 or yolov4')
parser.add_argument('--conf_thres', type=float, default=0.15, help='define confidence threshold')
parser.add_argument('--iou_thres', type=float, default=0.6, help='define iou threshold')
parser.add_argument('--score', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--keras2torch_json',  type=str, default='./keras_2_torch_names.json', help='path name transaltion tabular json file')
parser.add_argument('--out_path',  type=str, default='./intuitus_commands/yolov3-tiny-commands', help='command output path')
parser.add_argument('--sim_img',  type=str, default='../zybo_yolo/cam_data/room_cl.npz', help='image for simulation')
parser.add_argument('--classes_path',  type=str, default='../zybo_yolo/data/classes/coco.names', help='path to class name file')
parser.add_argument('--intuitus_ip_path',  type=str, default='../Intuitus/Intuitus_1.0', help='path to Intuitus FPGA IP path.')
parser.add_argument('--testbench',  type=str, default='tb_data_streamer', help='testbench used for simulation. Options: "tb_intuitus", tb_data_streamer".')

flags=parser.parse_args()

out_path = pathlib.Path(flags.out_path).absolute()
# %% load pretrained model. Use example_nn.py for training the model
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = load_config(flags.model,flags.tiny)

input_layer = tf.keras.layers.Input([flags.input_size, flags.input_size, 3],name='input_0')
if flags.model == 'yolov3':
    if flags.tiny:
        bckbn = darknet53_tiny_folding_bn
    else:
        raise NotImplementedError("Yolov3 folding batch norm not implemented yet. Simply copy backbone and set bn to False.")
else:
    if flags.tiny:
        bckbn = cspdarknet53_tiny_folding_bn
    else:
        raise NotImplementedError("Yolov4 folding batch norm not implemented yet. Simply copy backbone and set bn to False.")        

feature_maps_fbn = YOLO(input_layer, NUM_CLASS, flags.model, flags.tiny, fbn=True)
bbox_tensors_fbn = []
prob_tensors_fbn = []   

if flags.tiny:
    for i, fm in enumerate(feature_maps_fbn):
        if i == 0:
            output_tensors_fbn = decode(fm, flags.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
        else:
            output_tensors_fbn = decode(fm, flags.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
        bbox_tensors_fbn.append(output_tensors_fbn[0])
        prob_tensors_fbn.append(output_tensors_fbn[1])          
      
else:
    for i, fm in enumerate(feature_maps_fbn):
        if i == 0:
            output_tensors_fbn = decode(fm, flags.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
        elif i == 1:
            output_tensors_fbn = decode(fm, flags.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')            
        else:
            output_tensors_fbn = decode(fm, flags.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
        bbox_tensors_fbn.append(output_tensors_fbn[0])
        prob_tensors_fbn.append(output_tensors_fbn[1])             
      
       
pred_bbox = tf.concat(bbox_tensors_fbn, axis=1)
pred_prob = tf.concat(prob_tensors_fbn, axis=1)
pred = (pred_bbox, pred_prob)
modelN = tf.keras.Model(input_layer, pred) 
modelN.summary()
convert_torch2qKeras_param(modelN, flags.weights, flags.model, flags.tiny, flags.keras2torch_json)

# %% Initialize Intuitus model using pretrained keras model
model = Intuitus_Converter(modelN,out_path=out_path)

# %% translate model and weights to intuitus interpretable commands
# command outputs are saved in outpath
commands = model.translate()

# %% numpy simulation 
if flags.sim_img != None:
    import intuitus_simulator as sim
    hw_path = pathlib.Path(sim.__file__).absolute().parents[2]  / "Intuitus_1.0" 
    sim_model = sim.Intuitus_Simulator(model)
    
    if '.npz' in flags.sim_img:
        img_npz = np.load(str(flags.sim_img),allow_pickle=True)
        image_data = img_npz['img']
    else:
        original_image = cv2.imread(flags.sim_img)
        image_data = cv2.resize(original_image, (flags.input_size, flags.input_size)) 
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    sim_layers = sim_model.get_layers()
    start = timeit.default_timer()
    layer_nbr = 21
    #fmap = np.clip(np.round(imgs*2**7),(-1)*(2**7-1),2**7-1) 
    
    sim_layer_outputs = [image_data.reshape((1,)+image_data.shape)]
    for i in range(layer_nbr):
        if isinstance(sim_layers[i+1],tf.keras.layers.MaxPooling2D):
            stride = sim_layers[i+1].strides[0]
            sim_layer_outputs.append(sim_model.max_pool_2d(sim_layer_outputs[i],stride,True))
        elif 'resize' in sim_layers[i+1].name or 'Resize' in sim_layers[i+1].name:
            sim_layer_outputs.append(sim_model.upsample(sim_layer_outputs[i],True))
        elif 'concat' in sim_layers[i+1].name:
            sim_layer_outputs.append(np.concatenate((sim_layer_outputs[i],sim_layer_outputs[i-7]),axis=-1))
        else:
            if i == 18:
                sim_layer_outputs.append(sim_layers[i+1].sim_hw(sim_layer_outputs[14]))
            elif i == 19:
                sim_layer_outputs.append(sim_layers[i+1].sim_hw(sim_layer_outputs[18]))
            elif i == 20:
                sim_layer_outputs.append(sim_layers[i+1].sim_hw(sim_layer_outputs[19]))                
            else:
                sim_layer_outputs.append(sim_layers[i+1].sim_hw(sim_layer_outputs[i]))
    stop = timeit.default_timer()
    #test_sim = sim_layers[3].sim_hw_1(sim_layer_outputs[2])
    print('Runtime Numpy implementation: ', stop - start)  
# %% Yolo Layer 
    result_scale = 2.0**-4.0
    yolo_config = sim.special.yolov3_tiny_config()    
    Yolo_lb = sim.special.YoloLayer(yolo_config['lb']['anchors'],
                        yolo_config['lb']['classes'],
                        yolo_config['lb']['stride'])
    Yolo_mb = sim.special.YoloLayer(yolo_config['mb']['anchors'],
                        yolo_config['mb']['classes'],
                        yolo_config['mb']['stride'])    
    
    pred_lbbox = Yolo_lb(np.moveaxis(sim_layer_outputs[21][0,...],-1,0)*result_scale)
    pred_mbbox = Yolo_mb(np.moveaxis(sim_layer_outputs[20][0,...],-1,0)*result_scale)
    inf_out = np.concatenate((pred_lbbox,pred_mbbox),axis=0)
    boxes, pred_conf, classes = sim.special.filter_boxes(inf_out,flags.conf_thres)
    best_bboxes = sim.special.nms(boxes, pred_conf, classes, iou_threshold = flags.iou_thres, 
                      score=flags.score,method='merge')
    classes = sim.special.read_class_names(flags.classes_path)
    image = sim.special.draw_bbox_nms(np.uint8(image_data), best_bboxes,classes)
    plt.imshow(image)
    
# %% hardware simulation 
if flags.intuitus_ip_path != None and flags.sim_img != None and flags.testbench != None:
    layer_nbr = 1
    command = 0
    sim_model.run_hw_sim(sim_layer_outputs[layer_nbr-1], layer_nbr, pathlib.Path(flags.intuitus_ip_path), 
                         max_pooling=0, commands = commands[command], testbench=flags.testbench,
                         max_in_channels=5, start_tile=0, max_tiles = 3, waveform=True)
# %% software simulation   
