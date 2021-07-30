# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:28:22 2021

@author: lukas
"""

import tensorflow_model_optimization as tfmot
import pathlib
import shutil

import tensorflow as tf
import pathlib

from tensorflow import keras

from intuitus_converter import Intuitus_Converter
from Intuitus.core import float8, float6

#%% Parameter 
MODEL_NAME = 'MnistNN' # don't forget to edit function name of model
out_path = pathlib.Path('./mnist_command_out')
nb_epoch = 10
batch_size = 128
model_path = pathlib.Path(__file__).parent / MODEL_NAME

if out_path.exists():
    shutil.rmtree(out_path, ignore_errors=False, onerror=None)    
out_path.mkdir(parents=True, exist_ok=True)

#%% Load dataset
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

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

train_images_subset = train_images[0:1000] # out of 60000
train_labels_subset = train_labels[0:1000]

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=1, validation_split=0.1)

_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

fpga_model = Intuitus_Converter(model,out_path=out_path)
fpga_model.keras_model.set_weights(fpga_model.quantize_weights_and_bias(q_aware_model.get_weights())) 
fpga_model.translate()