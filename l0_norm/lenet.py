import tensorflow as tf
from layer import dense_l0


class lenet300:
    def __init__(self, phase='TRAIN', init_dropout_ratio=0.5, beta=0.66667, gamma=-0.1, zeta=1.1, lambdaa=0.0001):
        self.init_dropout_ratio = init_dropout_ratio
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.lambdaa = lambdaa

        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.h1, self.penalty1 = dense_l0('layer1', self.x, phase, 300, activation_fn=tf.nn.relu)
        self.h2, self.penalty2 = dense_l0('layer2', self.h1, phase, 100, activation_fn=tf.nn.relu)
        self.y_logit, self.penalty3 = dense_l0('layer3', self.h2, phase, 10)
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        with tf.name_scope('loss'):
            self.batch_size = tf.cast(tf.shape(self.x, out_type=tf.int32)[0], tf.float32)
            self.ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_logit)) / self.batch_size
            self.p1_loss = tf.reduce_sum(self.penalty1) / self.batch_size
            self.p2_loss = tf.reduce_sum(self.penalty2) / self.batch_size
            self.p3_loss = tf.reduce_sum(self.penalty3) / self.batch_size
            self.loss = self.ce_loss + self.lambdaa * (self.p1_loss + self.p2_loss + self.p3_loss)
        with tf.name_scope('summary'):
            tf.summary.scalar('cross_entropy', self.ce_loss)
            tf.summary.scalar('penalty1', self.p1_loss)
            tf.summary.scalar('penalty2', self.p2_loss)
            tf.summary.scalar('penalty3', self.p3_loss)
            tf.summary.histogram('penalty1_dist', self.penalty1)
            tf.summary.histogram('penalty2_dist', self.penalty2)
            tf.summary.histogram('penalty3_dist', self.penalty3)
            tf.summary.scalar('loss', self.loss)
            self.summary = tf.summary.merge_all()
        with tf.name_scope('prune'):
            self.threshold = tf.placeholder(tf.float32, [], 'threshold')
            self.count1 = tf.reduce_sum(tf.cast(tf.greater(self.penalty1, self.threshold), tf.float32))
            self.count2 = tf.reduce_sum(tf.cast(tf.greater(self.penalty2, self.threshold), tf.float32))
            self.count3 = tf.reduce_sum(tf.cast(tf.greater(self.penalty3, self.threshold), tf.float32))
        with tf.name_scope('accuracy'):
            self.label = tf.arg_max(self.y, -1)
            self.label_hat = tf.arg_max(self.y_logit, -1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.label, tf.int32), tf.cast(self.label_hat, tf.int32)), tf.float32))
        with tf.name_scope('optimizer'):
            self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)

    def partial_train(self, x, y, lr, sess, writer, record=True):
        loss, _, summary = sess.run([self.loss, self.optimizer, self.summary], feed_dict={self.x: x, self.y: y, self.lr: lr})
        if record == True:
            writer.add_summary(summary, self.global_step.eval(sess))
        return loss

    def pruned_structure(self, sess, threshold=0.05):
        count1, count2, count3 = sess.run([self.count1, self.count2, self.count3], feed_dict={self.threshold: threshold})
        return count1, count2, count3

    def test_batch(self, x, y, sess):
        accuracy, l, l_hat = sess.run([self.accuracy, self.label, self.label_hat], feed_dict={self.x: x, self.y: y})
        return accuracy