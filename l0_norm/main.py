import tensorflow as tf 
from lenet import lenet300
import math
from dataset import shuffle_data, load_mnist_data
import os


def train_model(model, sess, writer, x, y, num_epoch, batch_size=100, lr=0.001):
    iteration_per_epoch = math.floor(int(x.shape[0] / batch_size))
    for epoch in range(num_epoch):
        x, y = shuffle_data(x, y)
        total_loss = 0
        for i in range(iteration_per_epoch):
            batch_loss = model.partial_train(x, y, lr, sess, writer)
            total_loss += batch_loss
        total_loss /= iteration_per_epoch
        print('Epoch = {0}, loss = {1}.'.format(epoch, total_loss))


def main():
    x_train, y_train = load_mnist_data('training')
    x_test, y_test = load_mnist_data('testing')

    model = lenet300()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('graph', sess.graph)
        sess.run(tf.global_variables_initializer())

        train_model(model, sess, writer, x_train, y_train, 200)

        count1, count2, count3 = model.pruned_structure()
        print('Prunced structure: {0}-{1}-{2}.'.format(count1, count2, count3))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
