import tensorflow as tf
from layer import dense_l0


class lenet300:
    def __init__(self, init_dropout_ratio=0.5, beta=0.66667, gamma=-0.1, zeta=1.1):
        self.init_dropout_ratio = init_dropout_ratio
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.h1, self.penalty1 = dense_l0('layer1', self.x, 300, activation_fn=tf.nn.relu)
        self.h2, self.penalty2 = dense_l0('layer2', h1, 100, activation_fn=tf.nn.relu)
        self.y_logit, self.penalty3 = dense_l0('layer3', h2, 10)
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_logit, name='loss')