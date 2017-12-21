import tensorflow as tf 
from tensorflow.contrib import layers
import math


PROTECTOR = 0.00001


def dense_l0(name, x, phase, output_channel, reg=None, bias=False, init_dropout_ratio=0.5, beta=2.0/3.0, gamma=-0.1, zeta=1.1, activation_fn=None):
    input_channel = int(x.get_shape()[-1])
    assert len(x.get_shape()) == 2
    beta_inv = 1.0 / beta
    bgz = -beta * math.log(-gamma / zeta)
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [input_channel, output_channel], tf.float32, layers.xavier_initializer(), reg)
#        log_alpha = tf.get_variable('log_alpha', [input_channel, 1], tf.float32, tf.random_normal_initializer(tf.log(init_dropout_ratio / (1.0 - init_dropout_ratio)), 0.01))
        log_alpha = tf.get_variable('log_alpha', [input_channel, 1], tf.float32, tf.random_normal_initializer(tf.log(init_dropout_ratio), 1.0, dtype=tf.float32))
        if bias:
            b = tf.get_variable('b', [output_channel], tf.float32, tf.zeros_initializer())
    with tf.name_scope(name):
        alpha = tf.maximum(PROTECTOR, tf.exp(log_alpha), 'alpha')
        if phase == 'TRAIN':
            u = tf.random_uniform([input_channel, 1], PROTECTOR, 1.0 - PROTECTOR, tf.float32, name='u')    
            s = tf.divide(1.0, 1.0 + tf.pow((1.0 - u)/u/alpha, beta_inv), name='s')
            s_bar = s * (zeta - gamma) + gamma
            z = tf.minimum(tf.maximum(s_bar, 0.0), 1.0, 'z')
            l0_penalty = tf.nn.sigmoid(log_alpha + bgz, 'l0_penalty')
        else:
            z = tf.minimum(tf.maximum(alpha/(1 + alpha)*(zeta - gamma) + gamma, 0.0), 1.0, 'z')
            l0_penalty = z
        z_tile = tf.tile(z, [1, output_channel], 'z_tile')
        w_gated = tf.multiply(w, z_tile, 'w_gated')
        if bias:
            y = tf.nn.bias_add(tf.matmul(x, w_gated), b, name='output')
        else:
            y = tf.matmul(x, w_gated, name='output')
        if activation_fn is not None:
            y = activation_fn(y, name='output_activate')
    return y, l0_penalty


def dense(name, x, output_channel, reg=None, bias=False, activation_fn=None):
    input_channel = int(x.get_shape())[-1]
    assert len(x.get_shape()) == 2
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [input_channel, output_channel], tf.float32, layers.xavier_initializer(), reg)
        if bias:
            b = tf.get_variable('b', [output_channel], tf.float32, tf.zeros_initializer())
    with tf.name_scope(name):
        if bias:
            y = tf.nn.bias_add(tf.matmul(x, w), b, name='output')
        else:
            y = tf.matmul(x, w, name='output')
        if activation_fn is not None:
            y = activation_fn(y, name='output_activate')
    return y