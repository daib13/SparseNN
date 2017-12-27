import tensorflow as tf 
from tensorflow.contrib import layers
import math
import numpy as np


PROTECTOR = 0.00001


def dense_l0(name, x, phase, output_channel, reg=None, bias=False, init_log_alpha=-2.3, beta=2.0/3.0, gamma=-0.1, zeta=1.1, activation_fn=None):
    input_channel = int(x.get_shape()[-1])
    assert len(x.get_shape()) == 2
    beta_inv = 1.0 / beta
    bgz = -beta * math.log(-gamma / zeta)
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [input_channel, output_channel], tf.float32, layers.xavier_initializer(), reg)
        log_alpha = tf.get_variable('log_alpha', [input_channel, 1], tf.float32, tf.random_normal_initializer(init_log_alpha, 0.01, dtype=tf.float32))
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


def conv_l0(name, x, phase, output_channel, kernel_size=3, reg=None, init_log_alpha=-2.3, beta=0.667, gamma=-0.1, zeta=1.1):
    input_channel = int(x.get_shape()[-1])
    assert len(x.get_shape()) == 4
    beta_inv = 1.0 / beta
    bgz = -beta * math.log(-gamma / zeta)
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [kernel_size, kernel_size, input_channel, output_channel], tf.float32, layers.xavier_initializer(), reg)
        log_alpha = tf.get_variable('log_alpha', [output_channel], tf.float32, tf.random_normal_initializer(init_log_alpha, 0.01, dtype=tf.float32))
        b = tf.get_variable('b', [output_channel], tf.float32, tf.zeros_initializer())
    with tf.name_scope(name):
        alpha = tf.maximum(PROTECTOR, tf.exp(log_alpha), 'alpha')
        if phase == 'TRAIN':
            u = tf.random_uniform([output_channel], PROTECTOR, 1.0 - PROTECTOR, tf.float32, name='u')
            s = tf.divide(1.0, 1.0 + tf.pow((1.0 - u) / u / alpha, beta_inv), name='s')
            s_bar = s * (zeta - gamma) + gamma
            z = tf.minimum(tf.maximum(s_bar, 0.0), 1.0, 'z')
            l0_penalty = tf.nn.sigmoid(log_alpha + bgz, 'l0_penalty')
        else:
            z = tf.minimum(tf.maximum(alpha/(1 + alpha)*(zeta - gamma) + gamma, 0.0), 1.0, 'z')
            l0_penalty = z
        z_tile = tf.tile(tf.reshape(z, [1, 1, 1, output_channel]), [kernel_size, kernel_size, input_channel, 1], 'z_tile')
        w_gated = tf.multiply(w, z_tile, 'w_gated')
        b_gated = tf.multiply(b, z, 'b_gated')
        y = tf.nn.bias_add(tf.nn.conv2d(x, w_gated, [1, 1, 1, 1], 'SAME'), b_gated, name='output')
    return y, l0_penalty


def conv_bn_relu(name, x, output_channel, phase='TRAIN', reg=None, dropout=None):
    is_train = (phase == 'TRAIN')
    input_channel = int(x.get_shape()[-1])
    assert len(x.get_shape()) == 4
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [3, 3, input_channel, output_channel], tf.float32, layers.xavier_initializer(), reg)
        b = tf.get_variable('b', [output_channel], tf.float32, tf.zeros_initializer())
    with tf.name_scope(name):
        y = tf.nn.bias_add(tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME'), b, name='conv')
        y = layers.batch_norm(y)
        y = tf.nn.relu(y, 'relu')
        if dropout is not None:
            if phase == 'TRAIN':
                y = tf.nn.dropout(y, 1.0 - dropout, name='dropout')
            else:
                y = tf.multiply(y, dropout, 'dropout')
    return y


def conv_l0_bn_relu(name, x, output_channel, phase='TRAIN', reg=None, init_log_alpha=-2.3):
    is_train = (phase == 'TRAIN')
    y, l0_penalty = conv_l0(name + '/conv', x, phase, output_channel, kernel_size=3, reg=reg, init_log_alpha=init_log_alpha)
    with tf.name_scope(name):
        y = layers.batch_norm(y)
        y = tf.nn.relu(y, 'relu')
    return y, l0_penalty


def fc_bn_relu(name, x, output_channel, phase='TRAIN', reg=None, dropout=None):
    is_train = (phase == 'TRAIN')
    input_channel = int(x.get_shape()[-1])
    assert len(x.get_shape()) == 2
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [input_channel, output_channel], tf.float32, layers.xavier_initializer(), reg)
        b = tf.get_variable('b', [output_channel], tf.float32, tf.zeros_initializer())
    with tf.name_scope(name):
        y = tf.nn.bias_add(tf.matmul(x, w), b, name='fc')
        y = layers.batch_norm(y)
        y = tf.nn.relu(y, 'relu')
        if dropout is not None:
            if phase == 'TRAIN':
                y = tf.nn.dropout(y, 1.0 - dropout, name='dropout')
            else:
                y = tf.multiply(y, dropout, 'dropout')
    return y


def fc_l0_bn_relu(name, x, output_channel, phase='TRAIN', reg=None, init_log_alpha=-2.3):
    is_train = (phase == 'TRAIN')
    y, l0_penalty = dense_l0(name + '/fc', x, phase, output_channel, reg, True, init_log_alpha)
    with tf.name_scope(name):
        y = layers.batch_norm(y)
        y = tf.nn.relu(y, 'relu')
    return y, l0_penalty


class dense_layer:
    def __init__(self, name, x, phase, layer_type, output_channel, reg=None, bias=False, init_log_alpha=0.1, beta=0.667, gamma=-0.1, zeta=1.1, activation_fn=None):
        self.input_channel = int(x.get_shape()[-1])
        assert len(x.get_shape()) == 2
        self.name = name
        self.x = x
        self.layer_type = layer_type
        self.output_channel = output_channel
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.activation_fn = activation_fn
        self.beta_inv = 1.0 / beta
        self.bgz = -beta * math.log(-gamma / zeta)
        self.all_weights = []
        self.trainable_weights = []
        with tf.variable_scope(name + '_w'):
            self.w = tf.get_variable('w', [self.input_channel, self.output_channel], tf.float32, layers.xavier_initializer(), reg)
            self.trainable_weights.append(self.w)
            self.all_weights.append(self.w)
            if bias:
                self.b = tf.get_variable('b', [self.output_channel], tf.float32, tf.zeros_initializer())
                self.trainable_weights.append(self.b)
                self.all_weights.append(self.b)
            self.log_alpha = tf.get_variable('log_alpha', [self.input_channel, 1], tf.float32, tf.random_normal_initializer(init_log_alpha, 0.01, dtype=tf.float32))
            if layer_type == 'STOCHASTIC':
                self.trainable_weights.append(self.log_alpha)
            self.all_weights.append(self.log_alpha)
        with tf.name_scope(name):
            if self.layer_type == 'DETERMINISTIC':
                self.y = tf.matmul(self.x, self.w)
                self.l0_penalty = tf.constant(np.ones([self.input_channel, 1]), tf.float32, [self.input_channel, 1], name='l0_penalty')
            else:
                self.alpha = tf.maximum(PROTECTOR, tf.exp(self.log_alpha), 'alpha')
                if phase == 'TRAIN':
                    u = tf.random_uniform([self.input_channel, 1], PROTECTOR, 1.0 - PROTECTOR, tf.float32, name='u')
                    s = tf.divide(1.0, 1.0 + tf.pow((1.0 - u)/u/self.alpha, self.beta_inv), name='s')
                    s_bar = s * (self.zeta - self.gamma) + self.gamma
                    self.z = tf.minimum(tf.maximum(s_bar, 0.0), 1.0, 'z')
                    self.l0_penalty = tf.nn.sigmoid(self.log_alpha + self.bgz, 'l0_penalty')
                else:
                    self.z = tf.minimum(tf.maximum(self.alpha/(1 + self.alpha)*(self.zeta - self.gamma) + self.gamma, 0.0), 1.0, 'z')
                    self.l0_penalty = self.z
                z_tile = tf.tile(self.z, [1, self.output_channel], 'z_tile')
                w_gated = tf.multiply(self.w, z_tile, 'w_gated')
                self.y = tf.matmul(self.x, w_gated)
            if bias:
                self.y = self.y + self.b
            if self.activation_fn is not None:
                self.y = self.activation_fn(self.y)