from vgg import vgg, vgg_l0
from dataset import make_cifar10_dataset, shuffle_data
import math
import tensorflow as tf 
import os
import sys
from preprocess import preprocess, normalize
import numpy as np


def test_model(model, x, y, sess):
    num_iteration = int(math.ceil(x.shape[0]/500))
    total_accuracy = 0.0
    for i in range(num_iteration):
        batch_x = x[i*500:(i+1)*500, :, :, :]
        batch_y = y[i*500:(i+1)*500, :]
        batch_accuracy = model.test(batch_x, batch_y, sess)
        total_accuracy += batch_accuracy
    total_accuracy /= num_iteration
    return total_accuracy


def train_model(model, x, x2, y, y2, sess, writer, num_epoch, flog, batch_size=100, lr=0.05):
    iteration_per_epoch = int(math.floor(x.shape[0]/batch_size))
    print('{0:8s}\t{1:8s}\t{2:8s}\t{3:8s}\t{4:8s}'.format('Epoch', 'Lr', 'Loss', 'Acc1', 'Acc2'))
    flog.write('{0:8s}\t{1:8s}\t{2:8s}\t{3:8s}\t{4:8s}\n'.format('Epoch', 'Lr', 'Loss', 'Acc1', 'Acc2'))
    for epoch in range(num_epoch):
        total_loss = 0
        x, y = shuffle_data(x, y)
        for i in range(iteration_per_epoch):
            batch_x = x[i*batch_size:(i+1)*batch_size, :, :, :]
            batch_y = y[i*batch_size:(i+1)*batch_size, :]
            batch_loss = model.partial_train(batch_x, batch_y, sess, writer, lr, i%10 == 0)
            total_loss += batch_loss
        total_loss /= iteration_per_epoch
        train_accuracy = model.test(x[0:100, :, :, :], y[0:100, :], sess)
        test_accuracy = model.test(x2[0:500, :, :, :], y2[0:500, :], sess)
        print('{0:8d}\t{1:8.4f}\t{2:8.4f}\t{3:8.4f}\t{4:8.4f}'.format(epoch, lr, total_loss, train_accuracy, test_accuracy))
        flog.write('{0:8d}\t{1:8.4f}\t{2:8.4f}\t{3:8.4f}\t{4:8.4f}\n'.format(epoch, lr, total_loss, train_accuracy, test_accuracy))
        if epoch % 30 == 29:
            lr *= 0.5
            train_accuracy = test_model(model, x, y, sess)
            test_accuracy = test_model(model, x2, y2, sess)
            print('{0:8s}\t{1:8s}\t{2:8s}\t{3:8.4f}\t{4:8.4f}'.format('--', '--', '--', train_accuracy, test_accuracy))
            flog.write('{0:8s}\t{1:8s}\t{2:8s}\t{3:8.4f}\t{4:8.4f}\n'.format('--', '--', '--', train_accuracy, test_accuracy))


def main(lambdaa=0.1, init_log_alpha=0.0):
    # load data and preprocess
    x_train, y_train, x_test, y_test = make_cifar10_dataset()
    x_train, data_mean, data_std = normalize(x_train)
    x_test = normalize(x_test, data_mean, data_std)
    print('mean = {0}\nstd={1}'.format(data_mean, data_std))

    if not os.path.exists('model'):
        os.mkdir('model')
    fid = open('result.txt', 'wt')
    flog = open('log.txt', 'wt')

    # train l0 pruned model
    model = vgg_l0('TRAIN', lambdaa, init_log_alpha)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph_l0', sess.graph)

        train_model(model, x_train, x_test, y_train, y_test, sess, writer, 50, flog)
        saver.save(sess, 'model/model_l0')

    # test l0 pruned model
    tf.reset_default_graph()
    model = vgg_l0('TEST')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'model/model_l0')
        train_accuracy = test_model(model, x_train, y_train, sess)
        test_accuracy = test_model(model, x_test, y_test, sess)
        print('Train accuracy = {0:8.4f}.\nTest accuracy = {1:8.4f}.'.format(train_accuracy, test_accuracy))
        flog.write('Train accuracy = {0:8.4f}.\nTest accuracy = {1:8.4f}.'.format(train_accuracy, test_accuracy))
        structure = model.pruned_structure(sess)
        for num_neuron in structure:
            fid.write('{0}\n'.format(num_neuron))
        fid.write('\n{0}\n{1}'.format(train_accuracy, test_accuracy))
    structure = [int(dim) for dim in structure]

    # train small model
    tf.reset_default_graph()
    model = vgg('TRAIN', structure)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph_small', sess.graph)

        train_model(model, x_train, x_test, y_train, y_test, sess, writer, 50, flog)
        saver.save(sess, 'model/model_small')

    # test small model
    tf.reset_default_graph()
    model = vgg('TEST', structure)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'model/model_small')

        train_accuracy = test_model(model, x_train, y_train, sess)
        test_accuracy = test_model(model, x_test, y_test, sess)
        print('Train accuracy = {0:8.4f}.\nTest accuracy = {1:8.4f}.'.format(train_accuracy, test_accuracy))
        flog.write('Train accuracy = {0:8.4f}.\nTest accuracy = {1:8.4f}.'.format(train_accuracy, test_accuracy))
        fid.write('\n\n{0:8.4f}\n{1:8.4f}'.format(train_accuracy, test_accuracy))
    fid.close()


def main2(if_bn=False):
    # load data and preprocess
    x_train, y_train, x_test, y_test = make_cifar10_dataset()
    x_train, data_mean, data_std = normalize(x_train)
    print('Mean = {0}, std = {1}.'.format(data_mean, data_std))
    x_test = normalize(x_test, data_mean, data_std)

    if not os.path.exists('model'):
        os.mkdir('model')
    fid = open('result.txt', 'wt')
    flog = open('log.txt', 'wt')

    # train l0 pruned model
    model = vgg('TRAIN', weight_decay=5e-4, if_bn=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph', sess.graph)

        train_model(model, x_train, x_test, y_train, y_test, sess, writer, 450, flog, batch_size=128)
        saver.save(sess, 'model/model')

        train_accuracy = test_model(model, x_train, y_train, sess)
        test_accuracy = test_model(model, x_test, y_test, sess)
        print('Train accuracy = {0:8.4f}.\nTest accuracy = {1:8.4f}.'.format(train_accuracy, test_accuracy))
        fid.write('{0:8.4f}\n{1:8.4f}'.format(train_accuracy, test_accuracy))

    # test l0 pruned model
    tf.reset_default_graph()
    model = vgg('TEST', if_bn=True)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'model/model')
        train_accuracy = test_model(model, x_train, y_train, sess)
        test_accuracy = test_model(model, x_test, y_test, sess)
        print('Train accuracy = {0:8.4f}.\nTest accuracy = {1:8.4f}.'.format(train_accuracy, test_accuracy))
        fid.write('{0:8.4f}\n{1:8.4f}'.format(train_accuracy, test_accuracy))
    fid.close()
    flog.close()


if __name__ == '__main__':
#    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#    main2()
    if len(sys.argv) == 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
        main2()
    elif len(sys.argv) == 4:
        lambdaa = float(sys.argv[1])
        init_log_alpha = tf.log(float(sys.argv[2]))
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
        main(lambdaa, init_log_alpha)
    else:
        print('Wrong arguments.\nExample: 1) python main_vgg.py gpu_id(run baseline)\tor\n2)python main_vgg.py lambda alpha gpu_id(run l0 regularization)')