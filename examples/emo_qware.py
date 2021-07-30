# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:23:26 2021

@author: lukas
"""

import tensorflow_model_optimization as tfmot
import pathlib
import numpy as np

import tensorflow as tf

from tensorflow import keras
import tensorflow_model_optimization as tfmot

#%% import custom modules 
import intuitus_converter.misc.util.optain_dataset as load 

#%% Parameter 
MODEL_NAME = 'ExampleNN' # don't forget to edit function name of model

nb_epoch = 3
batch_size = 256
filepath = pathlib.Path('C:/Users/lukas/Documents/SoC_Project/python/FER2013-database/fer2013/fer2013.csv')
model_path = pathlib.Path(__file__).parent / MODEL_NAME

# %% define emotions 
emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']

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

y_train_labels  = np.array([np.argmax(lst) for lst in y_train])
y_val_labels = np.array([np.argmax(lst) for lst in y_val])
y_test_labels = np.array([np.argmax(lst) for lst in y_test])

#%% Load model

model = keras.models.load_model(str(model_path / 'model.pb'))

#%% Quantize model

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()


q_aware_model.fit(X_train, y_train_labels,
                  batch_size=128, epochs=1, validation_data=(X_val, y_val_labels))

_, baseline_model_accuracy = model.evaluate(
    X_test, y_test, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   X_test, y_test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)