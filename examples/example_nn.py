# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:12:36 2020

@author: lukas
"""

#%% import public modules
import numpy as np 
import os, os.path
from tensorflow.keras import layers as keras_layer 
from tensorflow.keras import models
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow_model_optimization as tfmot
import tensorflow.keras.backend as keras_backend
import pathlib



#%% import custom modules 
import intuitus_converter.misc.util.optain_dataset as load 
import intuitus_converter.misc.util.visualize as vis
import intuitus_converter.misc.util.save_data as sd 
#%% Parameter 
MODEL_NAME = 'ExampleNN' # don't forget to edit function name of model

train_float = True
train_from_scratch = False
use_conv_weights_only = False
nb_epoch = 1
batch_size = 64
filepath = pathlib.Path('C:/Users/lukas/Documents/SoC_Project/python/FER2013-database/fer2013/fer2013.csv')
model_path = pathlib.Path(__file__).parent / MODEL_NAME
# %% create directory to save data
if not model_path.is_dir() :
    print('Create dir {}'.format(MODEL_NAME))
    os.mkdir(model_path)


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



# %% Print the shape of the dataset
print('Training dataset: \t{}'.format(X_train.shape) +'\t with {} emotions'.format(y_train.shape[1]))
print('Test dataset: \t \t{}'.format(X_test.shape) +'\t with {} emotions'.format(y_test.shape[1]))
print('Validation dataset: \t{}'.format(X_val.shape) +'\t with {} emotions'.format(y_val.shape[1]))

# %% Save loaded data
#sd.save_data(X_test, y_test,"_privatetest6_100pct")
# np.save( 'X_test_privatetest6_100pct', X_test)
# np.save( 'y_test_privatetest6_100pct', y_test)
# X_fname = 'X_test_privatetest6_100pct.npy'
# y_fname = 'y_test_privatetest6_100pct.npy'
# X = np.load(X_fname)
# y = np.load(y_fname)
print ('Private test set')
y_labels = [np.argmax(lst) for lst in y_test]
counts = np.bincount(y_labels)
labels = emo #['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#print (zip(labels, counts))

# %% plot test data 
#vis.overview(0,191, X_train)
#vis.show_single(X_train,1000)
# %% rename
y_train = y_train 
y_public = y_val 
y_private = y_test 
y_train_labels  = np.array([np.argmax(lst) for lst in y_train])
y_val_labels = np.array([np.argmax(lst) for lst in y_val])
y_test_labels = np.array([np.argmax(lst) for lst in y_test])

# %% create model 
keras_backend.set_floatx('float32')
model_old = models.load_model(str(model_path / 'model.pb'))

print('Initialize model..') 
input_shape=(48, 48, 1)   
input_img = keras_layer.Input(shape=input_shape)
conv1 = keras_layer.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(input_img)
conv2 = keras_layer.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(conv1)
conv3 = keras_layer.Conv2D(32, (3, 3),  strides=(2,2), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(conv2)
#pool4 = keras_layer.MaxPooling2D(pool_size=(2, 2))(conv3)

conv5 = keras_layer.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(conv3)
conv6 = keras_layer.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(conv5)
conv7 = keras_layer.Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(conv6)
#pool8 = keras_layer.MaxPooling2D(pool_size=(2, 2))(conv7)

conv9 = keras_layer.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(conv7)
conv10 = keras_layer.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(conv9)
conv11 = keras_layer.Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu', kernel_constraint=max_norm(0.5))(conv10)
#pool12 = keras_layer.MaxPooling2D(pool_size=(2, 2))(conv11)

flatten = keras_layer.Flatten()(conv11)  # this converts our 3D feature maps to 1D feature vectors
dense13 = keras_layer.Dense(64, activation='relu')(flatten)
dropout1 = keras_layer.Dropout(0.2)(dense13)
dense14 = keras_layer.Dense(64, activation='relu')(dense13)
dropout2 = keras_layer.Dropout(0.2)(dense14)
dense15 = keras_layer.Dense(6, activation='softmax')(dense14)
modelN = models.Model(inputs=input_img, outputs=dense15)

if not train_from_scratch:
    if use_conv_weights_only:
        layersN = modelN.layers
        layers_old = model_old.layers    
        j = 1
        for layer in layers_old:
            if str(layer.__class__) != str(layers_old[1].__class__):
                continue
            layersN[j].set_weights(layer.get_weights())
            j += 1
    else:
        modelN.set_weights(model_old.get_weights()) 
    
# %% define optimizer 
print('Initialize Optimizer..')   
modelN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
# %% visualize model 
plot_model(modelN, to_file='{}/model.png'.format(MODEL_NAME), show_shapes=True, show_layer_names=True)
# %% Train the model
if train_float:
    print('Training..')   
    modelF = modelN.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size,
              validation_data=(X_val, y_val), shuffle=True, verbose=1)    
#%% Quantize model

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(modelN)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

q_aware_model.fit(X_train, y_train_labels, epochs=1, batch_size=batch_size,
          validation_data=(X_val, y_val_labels), shuffle=True, verbose=1)  
#q_aware_model.fit(X_train, y_train,
#                  batch_size=500, epochs=1, validation_data=(X_val, y_val))

_, baseline_model_accuracy = modelN.evaluate(
    X_test, y_test, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   X_test, y_test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

layersN = modelN.layers
layers_q = q_aware_model.layers
j = 1
for i in range(2,11):
    wb = layers_q[i].get_weights()
    w = [wb[1], wb[0]]
    layersN[j].set_weights(w)
    j += 1

# %% save model
models.save_model(modelN,str(model_path / 'model.pb'))
#modelN.save(MODEL_NAME)
# %% plot data distribution
vis.plot_distribution(y_train_labels, y_val_labels,['Train dataset', 'Public dataset'], \
                      emo,MODEL_NAME, ylims =[8000,1000]) 
    
# %% Evaluate model
acc = modelF.history['accuracy']
val_acc = modelF.history['val_accuracy']
loss = modelF.history['loss']
val_loss = modelF.history['val_loss']
epochs = range(len(acc))
vis.plot_evaluation(epochs,acc,val_acc,loss,val_loss,MODEL_NAME)

# %% evaluate model on private test set
score = modelN.evaluate(X_test, y_test, verbose=0)
print ("model %s: %.2f%%" % (modelN.metrics_names[1], score[1]*100))

# %% prediction and true labels
y_prob = modelN.predict(X_test, batch_size=32, verbose=0)
y_pred = [np.argmax(prob) for prob in y_prob]
y_true = [np.argmax(true) for true in y_test]

# %% more plots
#vis.plot_subjects_with_probs(X,0, 36, y_prob,y_pred, y_true,labels,MODEL_NAME)
#vis.plot_distribution2(y_true, y_pred,labels,MODEL_NAME)
#vis.plot_confusion_matrix(y_true, y_pred,labels,MODEL_NAME)