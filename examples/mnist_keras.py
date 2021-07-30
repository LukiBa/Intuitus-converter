# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:41:57 2021

@author: lukas
"""

import tempfile
import os
import pathlib

import tensorflow as tf

from tensorflow import keras

#%% Parameter 
MODEL_NAME = 'MnistNN' # don't forget to edit function name of model

nb_epoch = 10
batch_size = 128
model_path = pathlib.Path(__file__).parent / MODEL_NAME

#%% Load dataset
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0
#%% create model 
# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
  keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2,2),padding='same', activation='relu'),
  keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
  keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2),padding='same', activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#%% train model 
model.fit(
  train_images,
  train_labels,
  epochs=nb_epoch, 
  batch_size=batch_size,
  validation_split=0.1,
  shuffle=True, 
  verbose=1
)
#%% save model 
keras.models.save_model(model,str(model_path / 'model.pb'))