import tensorflow as tf 
from tensorflow.contrib import layers
from layer import dense, klloss


class SparseVae:
    def __init__(self, input_dim, latent_dim, encoder_dim, decoder_dim, activation_fn=tf.nn.relu, weight_decay=None, data_type='BINARY', kl_type='Jeffery', phase='Train'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dim = encoder_dim
        self.encoder_num = len(encoder_dim)
        self.decoder_dim = decoder_dim
        self.decoder_num = len(decoder_dim)
        self.reg = None
        if weight_decay is not None:
            self.reg = layers.l2_regularizer(weight_decay)
        self.activation_fn = activation_fn
        self.kl_type = kl_type

        with tf.name_scope('x'):
            self.x = tf.placeholder(tf.float32, [None, input_dim], 'x')
            self.batch_size = tf.cast(tf.shape(self.x)[0], tf.float32, name='batch_size')
        h = self.x 
        for i in range(self.encoder_num):
            h = dense('encoder' + str(i), h, self.encoder_dim[i], reg=self.reg, activation_fn=self.activation_fn)
        with tf.name_scope('latent'):
            self.mu_z = dense('mu_z', h, self.latent_dim, reg=self.reg)
            self.logsd_z = dense('logsd_z', h, self.latent_dim, reg=self.reg)
            self.sd_z = tf.exp(self.logsd_z, 'sd_z')
        with tf.name_scope('sample'):
            if phase == 'Train':
                self.epsilon = tf.random_normal(tf.shape(self.mu_z), name='epsilon')
                self.z = tf.add(self.mu_z, self.sd_z * self.epsilon, 'z')
                h = self.z
            else:
                self.epsilon = tf.placeholder(tf.float32, [None, self.latent_dim], 'epsilon')
                h = self.epsilon
        for i in range(self.decoder_num):
            h = dense('decoder' + str(i), h, self.decoder_dim[i], reg=self.reg, activation_fn=self.activation_fn)
        self.x_hat_logit = dense('x_hat', h, input_dim, reg=self.reg)
        if data_type == 'BINARY':
            self.x_hat = tf.nn.sigmoid(self.x_hat_logit, 'x_hat')
        else:
            self.x_hat = self.x_hat_logit

        with tf.name_scope('loss'):
            self.kl_div = klloss('kl_div', self.mu_z, self.logsd_z, self.sd_z, self.kl_type)
            self.kl_loss = tf.reduce_mean(self.kl_div)
            tf.summary.scalar('kl_loss', self.kl_loss)
            if data_type == 'BINARY':
                self.gen_dis = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.x_hat_logit, name='l2_dis'), -1)
                self.gen_loss = tf.reduce_mean(self.gen_dis)
            else:
                self.log_gamma = tf.get_variable('log_gamma', [], tf.float32, tf.zeros_initializer(), self.reg)
                self.gamma = tf.exp(self.log_gamma)
                self.l2_dis = tf.square(self.x - self.x_hat)
                self.l2_loss = tf.reduce_mean(tf.reduce_sum(self.l2_dis, -1))
                tf.summary.scalar('l2_distance', self.l2_loss)
                self.gen_dis = tf.reduce_sum((self.l2_dis / self.gamma  - self.log_gamma) / 2.0, -1)
                self.gen_loss = tf.reduce_mean(self.gen_dis)
            self.loss = self.kl_loss + self.gen_loss
            tf.summary.scalar('loss', self.loss)
            self.summary = tf.summary.merge_all()

        with tf.name_scope('optimizer'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)

    def partial_train(self, sess, writer, x, lr, record=True):
        loss, _, summary = sess.run([self.loss, self.optimizer, self.summary], feed_dict={self.x: x, self.lr: lr})
        if record:
            writer.add_summary(summary, self.global_step.eval(sess))
        return loss

    def extract_latent(self, sess, x):
        mu_z, sd_z = sess.run([self.mu_z, self.sd_z], feed_dict={self.x: x})
        return mu_z, sd_z
