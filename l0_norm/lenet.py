import tensorflow as tf
from layer import dense_l0


class lenet300:
    def __init__(self, init_dropout_ratio=0.5, beta=0.66667, gamma=-0.1, zeta=1.1, lambdaa=0.0001):
        self.init_dropout_ratio = init_dropout_ratio
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.lambdaa = lambdaa

        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.h1, self.penalty1, self.log_alpha1 = dense_l0('layer1', self.x, 300, activation_fn=tf.nn.relu)
        self.h2, self.penalty2, self.log_alpha2 = dense_l0('layer2', self.h1, 100, activation_fn=tf.nn.relu)
        self.y_logit, self.penalty3, self.log_alpha3 = dense_l0('layer3', self.h2, 10)
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        with tf.name_scope('loss'):
            self.batch_size = tf.shape(self.x, tf.float32)[0]
            self.ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_logit)) / self.batch_size
            self.p1_loss = tf.reduce_sum(self.penalty1) / self.batch_size
            self.p2_loss = tf.reduce_sum(self.penalty2) / self.batch_size
            self.p3_loss = tf.reduce_sum(self.penalty3) / self.batch_size
            self.loss = self.ce_loss + self.lambdaa * (self.p1_loss + self.p2_loss + self.p3_loss)
        with tf.name_scope('prune'):
            self.threshold = tf.placeholder(tf.float32, [], 'threshold')
            self.count1 = tf.reduce_sum(tf.cast(tf.less(self.log_alpha1, self.threshold, 'count1'), tf.int32))
            self.count2 = tf.reduce_sum(tf.cast(tf.less(self.log_alpha2, self.threshold, 'count2'), tf.int32))
            self.count3 = tf.reduce_sum(tf.cast(tf.less(self.log_alpha3, self.threshold, 'count3'), tf.int32))
        with tf.name_scope('summary'):
            tf.summary.scalar('cross_entropy', self.ce_loss)
            tf.summary.scalar('penalty1', self.p1_loss)
            tf.summary.scalar('penalty2', self.p2_loss)
            tf.summary.scalar('penalty3', self.p3_loss)
            tf.summary.histogram('log_alpha1', self.log_alpha1)
            tf.summary.histogram('log_alpha2', self.log_alpha2)
            tf.summary.histogram('log_alpha3', self.log_alpha3)
            self.summary = tf.summary.scalar('loss', self.loss)
        with tf.name_scope('optimizer'):
            self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)

    def partial_train(self, x, y, lr, sess, writer, record=True):
        loss, _, summary = sess.run([self.loss, self.optimizer, self.summary], feed_dict={self.x: x, self.y: y, self.lr: lr})
        if record == True:
            writer.add_summary(summary, self.global_step.eval(sess))
        return loss

    def pruned_structure(self, sess, threshold=3.0):
        count1, count2, count3 = sess.run([self.count1, self.count2, self.count3], feed_dict={self.threshold: threshold})
        return count1, count2, count3
