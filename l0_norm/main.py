import tensorflow as tf 
from lenet import lenet300
import math
from dataset import shuffle_data, load_mnist_data
import os


def train_model(model, sess, writer, x, y, num_epoch, batch_size=100, lr=0.001):
    iteration_per_epoch = int(math.floor(x.shape[0] / batch_size))
    for epoch in range(num_epoch):
        x, y = shuffle_data(x, y)
        total_loss = 0
        for i in range(iteration_per_epoch):
            x_batch = x[i*batch_size:(i+1)*batch_size, :]
            y_batch = y[i*batch_size:(i+1)*batch_size, :]
            batch_loss = model.partial_train(x_batch, y_batch, lr, sess, writer)
            total_loss += batch_loss
        total_loss /= iteration_per_epoch
        accuracy = model.test_batch(x[0:batch_size, :], y[0:batch_size, :], sess)
        print('Epoch = {0}, loss = {1}, accuracy = {2}.'.format(epoch, total_loss, accuracy))


def main():
    x_train, y_train = load_mnist_data('training')
    x_test, y_test = load_mnist_data('testing')

    model = lenet300('TRAIN', 0.8)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph', sess.graph)
        sess.run(tf.global_variables_initializer())

        train_model(model, sess, writer, x_train, y_train, 200)
        saver.save(sess, 'model/model')

    tf.reset_default_graph()
    model = lenet300('TEST')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'model/model.ckpt')
        count1, count2, count3 = model.pruned_structure(sess)
        print('Prunced structure: {0}-{1}-{2}.'.format(count1, count2, count3))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
