# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:49:32 2020

@author: lukas
"""
#%% import public modules
import numpy as np
import matplotlib.pyplot as plt 
import pathlib
import timeit
from tensorflow.keras import optimizers
from tensorflow.keras import layers as keras_layer 
from tensorflow.keras import models as keras_models
import tensorflow.keras.backend as keras_backend

#%% import custom modules 
import intuitus_converter.misc.util.optain_dataset as load
from intuitus_converter import Intuitus_Converter
from intuitus_converter.keras import Conv2D_fl8, Conv2D_int8
from intuitus_converter.core import float8, float6, MACC, float12
from intuitus_simulator import Intuitus_Simulator
#import IntuitusExtension as C_impl
#%% Parameter 
MODEL_NAME = 'ExampleNN' # don't forget to edit function name of model
filepath = pathlib.Path(__file__).absolute().parents[2] / 'FER2013-database/fer2013/fer2013.csv'
model_path = pathlib.Path(__file__).absolute().parent / MODEL_NAME
hw_path = pathlib.Path(__file__).absolute().parents[2] / "Intuitus/Intuitus_1.0" 
out_path = model_path /'command_out'

nb_epoch = 2
batch_size = 128
train = False
evaluate_net = False
dsp_sim = False 
do_C = False
use_float8 = False

# %% define emotions 
emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']

# %% load pretrained model. Use example_nn.py for training the model
try: modelN = keras_models.load_model(str(model_path / 'model.pb'))
except:
    Exception('Create the model first!')
# %% Load test data     
try: 
    X_test = np.load(model_path / 'X_test.npy')
    y_test = np.load(model_path / 'Y_test.npy')
except: 
    X_test, y_test = load.load_data(sample_split=1.0,classes=emo,emotion=emotion,
                                    usage='PrivateTest',filepath=filepath)
    np.save(model_path /  'X_test', X_test)
    np.save(model_path /  'y_test', y_test)
try: 
    X_train = np.load(model_path / 'X_train.npy')
    y_train = np.load(model_path / 'y_train.npy')
except: 
    X_train, y_train = load.load_data(sample_split=1.0,classes=emo,emotion=emotion,
                                      usage= 'Training',filepath=filepath)
    np.save(model_path /  'X_train', X_train)
    np.save(model_path /  'y_train', y_train)
try: 
    X_val = np.load(model_path / 'X_val.npy')
    y_val = np.load(model_path / 'y_val.npy')
except: 
    X_val,y_val = load.load_data(sample_split=1.0,classes=emo,emotion=emotion,
                                 usage= 'PublicTest',filepath=filepath)
    np.save(model_path /  'X_val', X_val)
    np.save(model_path /  'y_val', y_val)
    
# %% Initialize Intuitus model using pretrained keras model
#keras_backend.set_floatx('float16')
if evaluate_net:
    score = modelN.evaluate(X_test, y_test, verbose=0)
    print ("model %s: %.2f%%" % (modelN.metrics_names[1], score[1]*100))
model = Intuitus_Converter(modelN,use_float8=use_float8)
if use_float8:
    model.keras_model.set_weights(model.quantize_weights_and_bias(model.get_weights())) 
if evaluate_net:
    score = model.evaluate(X_test, y_test, verbose=0)
    print ("model %s: %.2f%%" % (model.keras_model.metrics_names[1], score[1]*100))
# %% Do some training steps to show that training is still possible using the Intuitus model.
if train:
    modelF = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size,
              validation_data=(X_val, y_val), shuffle=True, verbose=1)   
    # %% evaluate model on private test set after training
    score = modelN.evaluate(X_test, y_test, verbose=0)
    print ("model %s: %.2f%%" % (modelN.metrics_names[1], score[1]*100))      
    # %% Quantize weights
    model.keras_model.set_weights(model.quantize_weights_and_bias(model.get_weights())) 
    # %% evaluate model on private test set after qunatizaton
    score = modelN.evaluate(X_test, y_test, verbose=0)
    print ("model %s: %.2f%%" % (modelN.metrics_names[1], score[1]*100))  
    # %% Save the training imporvement
    model.save_model(str(model_path / 'model.pb'))

# %% create an additonal model using intuitus layer (inherits from keas layers)
input_shape=(48, 48, 1)   
input_img = keras_layer.Input(shape=input_shape,dtype='float32')
conv1 = Conv2D_int8(16, (3, 3), strides=(1,1), padding='same', activation='relu',outshift=1,biasshift=1)(input_img)
conv2 = Conv2D_int8(32, (3, 3), padding='same', activation='relu',outshift=1,biasshift=1)(conv1)
conv3 = Conv2D_int8(32, (3, 3),  strides=(2,2), padding='same', activation='relu',outshift=1,biasshift=1)(conv2)
#pool4 = keras_layer.MaxPooling2D(pool_size=(2, 2))(conv3)

conv5 = Conv2D_int8(64, (3, 3), padding='same', activation='relu',outshift=0,biasshift=1)(conv3)
conv6 = Conv2D_int8(64, (3, 3), padding='same', activation='relu',outshift=0,biasshift=1)(conv5)
conv7 = Conv2D_int8(64, (3, 3), strides=(2,2), padding='same', activation='relu',outshift=0,biasshift=1)(conv6)
#pool8 = keras_layer.MaxPooling2D(pool_size=(2, 2))(conv7)

conv9 = Conv2D_int8(128, (3, 3), padding='same', activation='relu',outshift=0,biasshift=1)(conv7)
conv10 = Conv2D_int8(128, (3, 3), padding='same', activation='relu',outshift=0,biasshift=1)(conv9)
conv11 = Conv2D_int8(128, (3, 3), strides=(2,2), padding='same', activation='relu',outshift=0,biasshift=1)(conv10)
#pool12 = keras_layer.MaxPooling2D(pool_size=(2, 2))(conv11)

flatten = keras_layer.Flatten()(conv11)  # this converts our 3D feature maps to 1D feature vectors
dense13 = keras_layer.Dense(64, activation='relu')(flatten)
dropout1 = keras_layer.Dropout(0.2)(dense13)
dense14 = keras_layer.Dense(64, activation='relu')(dropout1)
dropout2 = keras_layer.Dropout(0.2)(dense14)
dense15 = keras_layer.Dense(6, activation='softmax')(dropout2)
modelSim = keras_models.Model(inputs=input_img, outputs=dense15)

# modelSim.layers[1].build(input_img.shape)
# modelSim.layers[2].build(conv1.shape)
# modelSim.layers[3].build(conv2.shape)
modelSim.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
weightlist = model.get_weights()
weightlist[0] = weightlist[0][:3,:3,:,:16]
weightlist[1] = weightlist[1][:16]
weightlist[2] = weightlist[2][:3,:3,:16,:]
modelSim.set_weights(weightlist) 
#modelSim.set_weights(model.get_weights()) 

#modelSim.layers[1].set_weights(model.keras_model.layers[1].get_weights()) 
#weightlist  = [np.random.rand(3,3,1,32),model.keras_model.layers[1].get_weights()[1] ]
#modelSim.layers[1].set_weights(weightlist) 
sim_model = Intuitus_Converter(modelSim,out_path,out_path=out_path)
sim_model = Intuitus_Simulator(sim_model)

# %% Keras Model Debug
layers = model.get_layers()
imgs = X_val[0:32,:,:,:]
inp = model.keras_model.input  # input placeholder
outputs = [layer.output for layer in layers]  # all layer outputs
#outputs = [outputs[i] for i in (1, 2, 3, 4, 6, 8)]  # remove dropout, reshape
functors = [keras_backend.function([inp], [out]) for out in outputs]  # evaluation functions
layer_outs = [func([imgs, 1.]) for func in functors]
# %% software simulation of data streamer  
commands = sim_model.translate() 
    

    # %% Do hardware simulation using numpy 
sim_layers = sim_model.get_layers()
start = timeit.default_timer()
layer_nbr = 1
if use_float8:
    fmap = float8(imgs)
else:
    fmap = np.clip(np.round(imgs*2**7),(-1)*(2**7-1),2**7-1) 
sim_layer_outputs = [fmap[1:2,:,:,:]]
for i in range(layer_nbr):
    sim_layer_outputs.append(sim_layers[i+1].sim_hw(sim_layer_outputs[i]))
stop = timeit.default_timer()
#test_sim = sim_layers[3].sim_hw_1(sim_layer_outputs[2])
print('Runtime Numpy implementation: ', stop - start)  

#%% new datastreamer 
commands = sim_model.translate_layer(1)
max_in_channels=128
max_tiles = 2000
fmap = sim_layer_outputs[0]
tiles= []        
for i in range(len(commands[2])):
    for j in range(commands[0][0].in_channels):
        if commands[3][i][4] > max_in_channels-1 or commands[3][i][3] > max_tiles-1:
            continue
        if use_float8:
            fmap = float8(fmap).to_binary()
        tile = fmap[0, commands[2][i][0]:commands[2][i][1],commands[2][i][2]:commands[2][i][3],j]
        
        tiles.append(tile)
data_streamer_test = sim_model.sim_data_streamer_int8(tiles[0],commands[0][0])
fmap_test = tiles[0]

# %% hardware simulation
#sim_model.keras_model.layers[1].strides = (2,2)
sim_model.run_hw_sim(sim_layer_outputs[layer_nbr-1], layer_nbr, hw_path, testbench='tb_intuitus',max_in_channels=8, max_tiles = 2000, waveform=True, use_float8 = use_float8)