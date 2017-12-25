import tensorflow as tf
from tensorflow.contrib import layers
from layer import conv_bn_relu, fc_bn_relu


class vgg:
    def __init__(self, phase='TRAIN'):
        self.phase = phase
        self.__build_network()

    def __build_network(self):
        with tf.name_scope('x'):
            self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'x')
            self.batch_size = tf.cast(tf.shape(self.x, out_type=tf.int32)[0], tf.float32)
        
        self.conv1_1 = conv_bn_relu('conv1_1', self.x, 64, self.phase, dropout=0.3)
        self.conv1_2 = conv_bn_relu('conv1_2', self.conv1_2, 64, self.phase)
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

        self.fc0 = tf.reshape(self.pool5, [self.batch_size, -1], name='fc0')
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
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.labels=self.y, logits=self.y_logit)) / self.batch_size
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_hat, self.y), tf.float32))

        self.summary = tf.summary.scalar('loss', self.loss)
        
        with tf.name_scope('optimizer'):
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.global_step = tf.get_variable('global_step', [], tf.float32, tf.zeros_initializer(), trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)

    def partial_train(self, x, y, sess, writer, lr, record=True):
        loss, _, summary = sess.run([self.loss, self.optimizer, self.summary], feed_dict={self.x: x, self.y: y, self.lr: lr})
        if record:
            writer.add_summary(summary, self.global_step.eval(sess))
        return loss

    def test(self, x, y, sess):
        accuracy = sess.run(self.accuracy, feed_dict={self.x: x, self.y: y})
        return accuracy