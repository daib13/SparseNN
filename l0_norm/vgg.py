import tensorflow as tf
from tensorflow.contrib import layers
from layer import conv_bn_relu, fc_bn_relu, conv_l0_bn_relu, fc_l0_bn_relu


class vgg:
    def __init__(self, phase='TRAIN'):
        self.phase = phase
        self.__build_network()

    def __build_network(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'x')
            self.batch_size = tf.shape(self.x, out_type=tf.int32)[0]
        
        self.conv1_1 = conv_bn_relu('conv1_1', self.x, 64, self.phase, dropout=0.3)
        self.conv1_2 = conv_bn_relu('conv1_2', self.conv1_1, 64, self.phase)
        self.pool1 = tf.nn.max_pool(self.conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1')

        self.conv2_1 = conv_bn_relu('conv2_1', self.pool1, 128, self.phase, dropout=0.4)
        self.conv2_2 = conv_bn_relu('conv2_2', self.conv2_1, 128, self.phase)
        self.pool2 = tf.nn.max_pool(self.conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool2')

        self.conv3_1 = conv_bn_relu('conv3_1', self.pool2, 256, self.phase, dropout=0.4)
        self.conv3_2 = conv_bn_relu('conv3_2', self.conv3_1, 256, self.phase, dropout=0.4)
        self.conv3_3 = conv_bn_relu('conv3_3', self.conv3_2, 256, self.phase)
        self.pool3 = tf.nn.max_pool(self.conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool3')

        self.conv4_1 = conv_bn_relu('conv4_1', self.pool3, 512, self.phase, dropout=0.4)
        self.conv4_2 = conv_bn_relu('conv4_2', self.conv4_1, 512, self.phase, dropout=0.4)
        self.conv4_3 = conv_bn_relu('conv4_3', self.conv4_2, 512, self.phase)
        self.pool4 = tf.nn.max_pool(self.conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool4')

        self.conv5_1 = conv_bn_relu('conv5_1', self.pool4, 512, self.phase, dropout=0.4)
        self.conv5_2 = conv_bn_relu('conv5_2', self.conv5_1, 512, self.phase, dropout=0.4)
        self.conv5_3 = conv_bn_relu('conv5_3', self.conv5_2, 512, self.phase)
        self.pool5 = tf.nn.max_pool(self.conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool5')

        self.fc0 = layers.flatten(self.pool5)
        if self.phase == 'TRAIN':
            self.fc0_dropout = tf.nn.dropout(self.fc0, 0.5, name='fc0_dropout')
        else:
            self.fc0_dropout = tf.multiply(self.fc0, 0.5, name='fc0_dropout')

        self.fc1 = fc_bn_relu('fc6', self.fc0_dropout, 512, self.phase, dropout=0.5)
        self.fc2 = fc_bn_relu('fc7', self.fc1, 512, self.phase, dropout=0.5)
        
        with tf.variable_scope('y_w'):
            w = tf.get_variable('w', [512, 10], tf.float32, layers.xavier_initializer())
            b = tf.get_variable('b', [10], tf.float32, tf.zeros_initializer())
        with tf.name_scope('y'):
            self.y_hat_logit = tf.nn.bias_add(tf.matmul(self.fc2, w), b, name='y_hat_logit')
            self.y_hat = tf.arg_max(self.y_hat_logit, -1, tf.int32, 'y_hat')
            self.y_logit = tf.placeholder(tf.float32, [None, 10], 'y_logit')
            self.y = tf.arg_max(self.y_logit, -1, tf.int32, 'y')
        with tf.name_scope('loss'):           
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_logit, logits=self.y_hat_logit)) / tf.cast(self.batch_size, tf.float32)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_hat, self.y), tf.float32))

        self.summary = tf.summary.scalar('loss', self.loss)
        
        with tf.name_scope('optimizer'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.global_step = tf.get_variable('global_step', [], tf.float32, tf.zeros_initializer(), trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def partial_train(self, x, y, sess, writer, lr, record=True):
        loss, _, summary = sess.run([self.loss, self.optimizer, self.summary], feed_dict={self.x: x, self.y_logit: y, self.lr: lr})
        if record:
            writer.add_summary(summary, self.global_step.eval(sess))
        return loss

    def test(self, x, y, sess):
        accuracy = sess.run(self.accuracy, feed_dict={self.x: x, self.y_logit: y})
        return accuracy


class vgg_l0:
    def __init__(self, phase='TRAIN', lambdaa=0.1, init_log_alpha=-2.3):
        self.phase = phase
        self.lambdaa = lambdaa
        self.init_log_alpha = init_log_alpha
        self.__build_network()

    def __build_network(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'x')
            self.batch_size = tf.shape(self.x, out_type=tf.int32)[0]

        self.conv1_1, self.l0_penalty_conv1_1 = conv_l0_bn_relu('conv1_1', self.x, 64, self.phase, init_log_alpha=self.init_log_alpha)
        self.conv1_2, self.l0_penalty_conv1_2 = conv_l0_bn_relu('conv1_2', self.conv1_1, 64, self.phase, init_log_alpha=self.init_log_alpha)
        self.pool1 = tf.nn.max_pool(self.conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1')

        self.conv2_1, self.l0_penalty_conv2_1 = conv_l0_bn_relu('conv2_1', self.pool1, 128, self.phase, init_log_alpha=self.init_log_alpha)
        self.conv2_2, self.l0_penalty_conv2_2 = conv_l0_bn_relu('conv2_2', self.conv2_1, 128, self.phase, init_log_alpha=self.init_log_alpha)
        self.pool2 = tf.nn.max_pool(self.conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool2')

        self.conv3_1, self.l0_penalty_conv3_1 = conv_l0_bn_relu('conv3_1', self.pool2, 256, self.phase, init_log_alpha=self.init_log_alpha)
        self.conv3_2, self.l0_penalty_conv3_2 = conv_l0_bn_relu('conv3_2', self.conv3_1, 256, self.phase, init_log_alpha=self.init_log_alpha)
        self.conv3_3, self.l0_penalty_conv3_3 = conv_l0_bn_relu('conv3_3', self.conv3_2, 256, self.phase, init_log_alpha=self.init_log_alpha)
        self.pool3 = tf.nn.max_pool(self.conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool3')

        self.conv4_1, self.l0_penalty_conv4_1 = conv_l0_bn_relu('conv4_1', self.pool3, 512, self.phase, init_log_alpha=self.init_log_alpha)
        self.conv4_2, self.l0_penalty_conv4_2 = conv_l0_bn_relu('conv4_2', self.conv4_1, 512, self.phase, init_log_alpha=self.init_log_alpha)
        self.conv4_3, self.l0_penalty_conv4_3 = conv_l0_bn_relu('conv4_3', self.conv4_2, 512, self.phase, init_log_alpha=self.init_log_alpha)
        self.pool4 = tf.nn.max_pool(self.conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool4')

        self.conv5_1, self.l0_penalty_conv5_1 = conv_l0_bn_relu('conv5_1', self.pool4, 512, self.phase, init_log_alpha=self.init_log_alpha)
        self.conv5_2, self.l0_penalty_conv5_2 = conv_l0_bn_relu('conv5_2', self.conv5_1, 512, self.phase, init_log_alpha=self.init_log_alpha)
        self.conv5_3, self.l0_penalty_conv5_3 = conv_l0_bn_relu('conv5_3', self.conv5_2, 512, self.phase, init_log_alpha=self.init_log_alpha)
        self.pool5 = tf.nn.max_pool(self.conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool5')

        self.fc0 = layers.flatten(self.pool5)

        self.fc1, self.l0_penalty_fc1 = fc_l0_bn_relu('fc6', self.fc0, 512, self.phase, init_log_alpha=self.init_log_alpha)
        self.fc2, self.l0_penalty_fc2 = fc_l0_bn_relu('fc7', self.fc1, 512, self.phase, init_log_alpha=self.init_log_alpha)
        
        with tf.variable_scope('y_w'):
            w = tf.get_variable('w', [512, 10], tf.float32, layers.xavier_initializer())
            b = tf.get_variable('b', [10], tf.float32, tf.zeros_initializer())
        with tf.name_scope('y'):
            self.y_hat_logit = tf.nn.bias_add(tf.matmul(self.fc2, w), b, name='y_hat_logit')
            self.y_hat = tf.arg_max(self.y_hat_logit, -1, tf.int32, 'y_hat')
            self.y_logit = tf.placeholder(tf.float32, [None, 10], 'y_logit')
            self.y = tf.arg_max(self.y_logit, -1, tf.int32, 'y')
        with tf.name_scope('loss'):
            self.l0_loss_conv1_1 = tf.reduce_sum(self.l0_penalty_conv1_1)
            self.l0_loss_conv1_2 = tf.reduce_sum(self.l0_penalty_conv1_2)
            self.l0_loss_conv2_1 = tf.reduce_sum(self.l0_penalty_conv2_1)
            self.l0_loss_conv2_2 = tf.reduce_sum(self.l0_penalty_conv2_2)
            self.l0_loss_conv3_1 = tf.reduce_sum(self.l0_penalty_conv3_1)
            self.l0_loss_conv3_2 = tf.reduce_sum(self.l0_penalty_conv3_2)
            self.l0_loss_conv3_3 = tf.reduce_sum(self.l0_penalty_conv3_3)
            self.l0_loss_conv4_1 = tf.reduce_sum(self.l0_penalty_conv4_1)
            self.l0_loss_conv4_2 = tf.reduce_sum(self.l0_penalty_conv4_2)
            self.l0_loss_conv4_3 = tf.reduce_sum(self.l0_penalty_conv4_3)
            self.l0_loss_conv5_1 = tf.reduce_sum(self.l0_penalty_conv5_1)
            self.l0_loss_conv5_2 = tf.reduce_sum(self.l0_penalty_conv5_2)
            self.l0_loss_conv5_3 = tf.reduce_sum(self.l0_penalty_conv5_3)
            self.l0_loss_fc1 = tf.reduce_sum(self.l0_penalty_fc1)
            self.l0_loss_fc2 = tf.reduce_sum(self.l0_penalty_fc2)

            self.l0_loss = (self.l0_loss_conv1_1 + self.l0_loss_conv1_2 + self.l0_loss_conv2_1 + self.l0_loss_conv2_2 + self.l0_loss_conv3_1 + self.l0_loss_conv3_2 + self.l0_loss_conv3_3 + self.l0_loss_conv4_1 + self.l0_loss_conv4_2 + self.l0_loss_conv4_3 + self.l0_loss_conv5_1 + self.l0_loss_conv5_2 + self.l0_loss_conv5_3 + self.l0_loss_fc1 + self.l0_loss_fc2) / tf.cast(self.batch_size, tf.float32)

            self.ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_logit, logits=self.y_hat_logit)) / tf.cast(self.batch_size, tf.float32)

            self.loss = self.l0_loss * self.lambdaa + self.ce_loss

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_hat, self.y), tf.float32))

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('ce_loss', self.ce_loss)
        tf.summary.scalar('l0_loss', self.l0_loss)
        tf.summary.scalar('conv1_1p', self.l0_loss_conv1_1)
        tf.summary.scalar('conv1_2p', self.l0_loss_conv1_2)
        tf.summary.scalar('conv2_1p', self.l0_loss_conv2_1)
        tf.summary.scalar('conv2_2p', self.l0_loss_conv2_2)
        tf.summary.scalar('conv3_1p', self.l0_loss_conv3_1)
        tf.summary.scalar('conv3_2p', self.l0_loss_conv3_2)
        tf.summary.scalar('conv3_3p', self.l0_loss_conv3_3)
        tf.summary.scalar('conv4_1p', self.l0_loss_conv4_1)
        tf.summary.scalar('conv4_2p', self.l0_loss_conv4_2)
        tf.summary.scalar('conv4_3p', self.l0_loss_conv4_3)
        tf.summary.scalar('conv5_1p', self.l0_loss_conv5_1)
        tf.summary.scalar('conv5_2p', self.l0_loss_conv5_2)
        tf.summary.scalar('conv5_3p', self.l0_loss_conv5_3)
        tf.summary.scalar('fc1p', self.l0_loss_fc1)
        tf.summary.scalar('fc2p', self.l0_loss_fc2)
        tf.summary.histogram('conv1_1h', self.l0_penalty_conv1_1)
        tf.summary.histogram('conv1_2h', self.l0_penalty_conv1_2)
        tf.summary.histogram('conv2_1h', self.l0_penalty_conv2_1)
        tf.summary.histogram('conv2_2h', self.l0_penalty_conv2_2)
        tf.summary.histogram('conv3_1h', self.l0_penalty_conv3_1)
        tf.summary.histogram('conv3_2h', self.l0_penalty_conv3_2)
        tf.summary.histogram('conv3_3h', self.l0_penalty_conv3_3)
        tf.summary.histogram('conv4_1h', self.l0_penalty_conv4_1)
        tf.summary.histogram('conv4_2h', self.l0_penalty_conv4_2)
        tf.summary.histogram('conv4_3h', self.l0_penalty_conv4_3)
        tf.summary.histogram('conv5_1h', self.l0_penalty_conv5_1)
        tf.summary.histogram('conv5_2h', self.l0_penalty_conv5_2)
        tf.summary.histogram('conv5_3h', self.l0_penalty_conv5_3)
        tf.summary.histogram('fc1h', self.l0_penalty_fc1)
        tf.summary.histogram('fc2h', self.l0_penalty_fc2)
        self.summary = tf.summary.merge_all()

        with tf.name_scope('optimizer'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.global_step = tf.get_variable('global_step', [], tf.float32, tf.zeros_initializer(), trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def partial_train(self, x, y, sess, writer, lr, record=True):
        loss, _, summary = sess.run([self.loss, self.optimizer, self.summary], feed_dict={self.x: x, self.y_logit: y, self.lr: lr})
        if record:
            writer.add_summary(summary, self.global_step.eval(sess))
        return loss

    def test(self, x, y, sess):
        accuracy = sess.run(self.accuracy, feed_dict={self.x: x, self.y_logit: y})
        return accuracy