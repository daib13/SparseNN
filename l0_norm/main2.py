import tensorflow as tf 
from lenet import lenet300_layerwise
import math
from dataset import shuffle_data, load_mnist_data
import os
import numpy as np


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


def test_model(model, sess, x, y):
    num_iteration = int(math.floor(x.shape[0] / 100))
    accuracy = []
    for i in range(num_iteration):
        accuracy.append(model.test_batch(x[i*100:(i+1)*100, :], y[i*100:(i+1)*100, :], sess))
    return np.mean(accuracy)


def main():
    x_train, y_train = load_mnist_data('training')
    x_test, y_test = load_mnist_data('testing')
        
    fid = open('accuracy_stage.txt', 'w')
    for stage in range(-1, 3):
        model = lenet300_layerwise('TRAIN', stage, math.log(0.1))

        with tf.Session() as sess:
            saver = tf.train.Saver(model.all_weights)
            writer = tf.summary.FileWriter('graph_stage', sess.graph)
            sess.run(tf.global_variables_initializer())
            if stage >= 0:
                saver.restore(sess, 'model/model_stage' + str(stage))

            train_model(model, sess, writer, x_train, y_train, 100)
            saver.save(sess, 'model/model_stage' + str(stage + 1))

        tf.reset_default_graph()
        
        model = lenet300_layerwise('TEST', stage)
        with tf.Session() as sess:
            saver = tf.train.Saver(model.all_weights)
            saver.restore(sess, 'model/model_stage' + str(stage + 1))
            count1, count2, count3 = model.pruned_structure(sess, 0.005)
            print('Prunced structure: {0}-{1}-{2}.'.format(count1, count2, count3))
            train_acc = test_model(model, sess, x_train, y_train)
            test_acc = test_model(model, sess, x_test, y_test)
            print('Train accuracy = {0}.'.format(train_acc))
            print('Test accuracy = {0}.'.format(test_acc))
            fid.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(count1, count2, count3, train_acc, test_acc))
        
        tf.reset_default_graph()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
