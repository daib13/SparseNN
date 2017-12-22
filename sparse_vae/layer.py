import tensorflow as tf 
from tensorflow.contrib import layers


PROTECTOR = 0.000001


def dense(name, x, output_channel, bias=True, reg=None, activation_fn=None):
    input_channel = int(x.get_shape()[-1])
    with tf.variable_scope(name + '_w'):
        w = tf.get_variable('w', [input_channel, output_channel], tf.float32, layers.xavier_initializer(), reg)
        if bias:
            b = tf.get_variable('b', [output_channel], tf.float32, tf.zeros_initializer(), reg)
    with tf.name_scope(name):
        if len(x.get_shape()) != 2:
            x = tf.reshape(x, [-1, input_channel], name='reshape')
        y = tf.matmul(x, w, name='prod')
        if bias:
            y = tf.nn.bias_add(y, b, name='sum')
        if activation_fn is not None:
            y = activation_fn(y, name='act')
    return y


def klloss(name, mu_z, logsd_z, sd_z, prior_type='Gaussian'):
    assert len(mu_z.get_shape()) == 2
    assert len(logsd_z.get_shape()) == 2
    assert len(sd_z.get_shape()) == 2
    if prior_type == 'Gaussian':
        with tf.name_scope(name):
            loss = tf.reduce_sum((tf.square(mu_z) + tf.square(sd_z) - 1.0) / 2.0 - logsd_z, -1)
    else:
        with tf.name_scope(name):
            latent_dim = int(mu_z.get_shape()[-1])
            s = tf.square(mu_z) + tf.square(sd_z)
            m = tf.reduce_mean(s, -1)
            loss = tf.log(tf.maximum(m, PROTECTOR)) * latent_dim / 2.0 - tf.reduce_sum(logsd_z, -1)
    return loss