import tensorflow as tf 
from tensorflow.contrib import layers


def dense_ib_layer(name, x, output_dim, phase, gamma, reg=None):
    input_dim = int(x.get_shape()[-1])
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [input_dim, output_dim], tf.float32, layers.xavier_initializer(), reg)
        b = tf.get_variable('b', [output_dim], tf.float32, tf.zeros_initializer())
        log_sigma2 = tf.get_variable('log_sigma2', [output_dim], tf.float32, tf.constant_initializer(-10.0))
    with tf.name_scope(name):
        y_mu = tf.nn.bias_add(tf.matmul(x, w_mu), b_mu, name='y_mu')
        y_sd = 