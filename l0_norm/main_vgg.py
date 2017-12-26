from vgg import vgg, vgg_l0
from dataset import make_cifar10_dataset, shuffle_data
import math
import tensorflow as tf 
import os
import sys


def train_model(model, x, y, sess, writer, num_epoch, batch_size=100, lr=0.001):
    iteration_per_epoch = int(math.floor(x.shape[0]/batch_size))
    for epoch in range(num_epoch):
        total_loss = 0
        shuffle_data(x, y)
        for i in range(iteration_per_epoch):
            batch_x = x[i*batch_size:(i+1)*batch_size, :, :, :]
            batch_y = y[i*batch_size:(i+1)*batch_size, :]
            batch_loss = model.partial_train(batch_x, batch_y, sess, writer, lr, i%10 == 0)
            total_loss += batch_loss
        total_loss /= iteration_per_epoch
        accuracy = model.test(x[0:100, :, :, :], y[0:100, :], sess)
        print('Epoch = {0}, loss = {1}, accuracy = {2}.'.format(epoch, total_loss, accuracy))


def test_model(model, x, y, sess):
    num_iteration = int(math.ceil(x.shape[0]/100))
    total_accuracy = 0.0
    for i in range(num_iteration):
        batch_x = x[i*100:(i+1)*100, :, :, :]
        batch_y = y[i*100:(i+1)*100, :]
        batch_accuracy = model.test(batch_x, batch_y, sess)
        total_accuracy += batch_accuracy
    total_accuracy /= num_iteration
    return total_accuracy


def main(lambdaa, init_log_alpha):
    x_train, y_train, x_test, y_test = make_cifar10_dataset()

    if not os.path.exists('model_vgg_l0'):
        os.mkdir('model_vgg_l0')

#    model = vgg_l0('TRAIN', lambdaa, init_log_alpha)
    model = vgg('TRAIN')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph_vgg_l0', sess.graph)

        train_model(model, x_train, y_train, sess, writer, 100)
        saver.save(sess, 'model_vgg_l0/model')

    tf.reset_default_graph()
#    model = vgg_l0('TEST')
    model = vgg('TEST')
#    fid = open('result.txt', 'wt')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'model_vgg_l0/model')

        train_accuracy = test_model(model, x_train, y_train, sess)
        test_accuracy = test_model(model, x_test, y_test, sess)
        print('Train accuracy = {0}.\nTest accuracy = {1}.'.format(train_accuracy, test_accuracy))
#        structure = model.pruned_structure(sess)
#        for num_neuron in structure:
#            fid.write('{0}\n'.format(num_neuron))
#        fid.write('\n{0}\n{1}'.format(train_accuracy, test_accuracy))
#    fid.close()


if __name__ == '__main__':
    lambdaa = float(sys.argv[1])
    init_log_alpha = math.log(float(sys.argv[2]))
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
    main(lambdaa, init_log_alpha)    