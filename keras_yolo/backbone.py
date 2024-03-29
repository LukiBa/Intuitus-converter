#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from .common import *

def darknet53(input_data):

    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)

    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def cspdarknet53(input_data):

    input_data = convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = convolutional(input_data, (1, 1, 2048, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data

def cspdarknet53_tiny(input_data):
    input_data = convolutional(0,input_data, (3, 3, 3, 32), downsample=True)
    input_data = convolutional(1,input_data, (3, 3, 32, 64), downsample=True)
    input_data = convolutional(2,input_data, (3, 3, 64, 64))

    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(3,input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = convolutional(4,input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(5,input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(6,input_data, (3, 3, 64, 128))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(7,input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = convolutional(8,input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(9,input_data, (1, 1, 64, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(10,input_data, (3, 3, 128, 256))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(11,input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = convolutional(12,input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(13,input_data, (1, 1, 128, 256))
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(14,input_data, (3, 3, 512, 512))

    return route_1, input_data

def darknet53_tiny(input_data):
    input_data = convolutional(0,input_data, (3, 3, 3, 16))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(1,input_data, (3, 3, 16, 32))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(2,input_data, (3, 3, 32, 64))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(3,input_data, (3, 3, 64, 128))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(4,input_data, (3, 3, 128, 256))
    route_1 = input_data
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(5,input_data, (3, 3, 256, 512))
    input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
    input_data = convolutional(6,input_data, (3, 3, 512, 1024))

    return route_1, input_data

def darknet53_tiny_folding_bn(input_data):
    input_data = convolutional(0,input_data, (3, 3, 3, 16), bn = False, quant = True)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(1,input_data, (3, 3, 16, 32), bn = False, quant = True)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(2,input_data, (3, 3, 32, 64), bn = False, quant = True)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(3,input_data, (3, 3, 64, 128), bn = False, quant = True)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(4,input_data, (3, 3, 128, 256), bn = False, quant = True, allow_immediate_maxpool = False)
    route_1 = input_data
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(5,input_data, (3, 3, 256, 512), bn = False, quant = True)
    input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
    input_data = convolutional(6,input_data, (3, 3, 512, 1024), bn = False, quant = True)

    return route_1, input_data

def cspdarknet53_tiny_folding_bn(input_data):
    id = 0
    input_data = convolutional(0,input_data, (3, 3, 3, 32), bn = False, downsample=True )
    input_data = convolutional(1,input_data, (3, 3, 32, 64), bn = False, downsample=True)
    input_data = convolutional(2,input_data, (3, 3, 64, 64), bn = False)

    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(3,input_data, (3, 3, 32, 32), bn = False)
    route_1 = input_data
    input_data = convolutional(4,input_data, (3, 3, 32, 32), bn = False)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(5,input_data, (1, 1, 32, 64), bn = False)
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(6,input_data, (3, 3, 64, 128), bn = False)
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(7,input_data, (3, 3, 64, 64), bn = False)
    route_1 = input_data
    input_data = convolutional(8,input_data, (3, 3, 64, 64), bn = False)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(9,input_data, (1, 1, 64, 128), bn = False)
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(10,input_data, (3, 3, 128, 256), bn = False)
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(11,input_data, (3, 3, 128, 128), bn = False)
    route_1 = input_data
    input_data = convolutional(12,input_data, (3, 3, 128, 128), bn = False)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(13,input_data, (1, 1, 128, 256), bn = False)
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(14,input_data, (3, 3, 512, 512), bn = False)

    return route_1, input_data
