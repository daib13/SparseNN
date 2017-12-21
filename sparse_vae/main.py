from vae import SparseVae
import tensorflow as tf 
import math
from dataset import load_mnist_data, shuffle_data
import numpy as np
import os
import matplotlib.pyplot as plt


def train_model(model, sess, writer, x, num_epoch, batch_size=100, lr=0.001):
    iteration_per_epoch = int(math.floor(x.shape[0] / batch_size))
    for epoch in range(num_epoch):
        total_loss = 0
        x = shuffle_data(x)
        for i in range(iteration_per_epoch):
            x_batch = x[i*batch_size:(i+1)*batch_size, :]
            batch_loss = model.partial_train(sess, writer, x_batch, lr)
            total_loss += batch_loss
        total_loss /= iteration_per_epoch
        print('Epoch = {0}, loss = {1}.'.format(epoch, total_loss))


def test_model(model, sess, x):
    batch_size = 100
    num_iteration = int(math.ceil(x.shape[0] / batch_size))
    mu_z = []
    sd_z = []
    for i in range(num_iteration):
        x_batch = x[i*batch_size:(i+1)*batch_size, :]
        batch_mu_z, batch_sd_z = model.extract_latent(sess, x_batch)
        mu_z.append(batch_size)
        sd_z.append(batch_sd_z)
    mu_z = np.concatenate(mu_z, 0)
    sd_z = np.concatenate(sd_z, 0)
    mu_z = mu_z[0:x.shape[0], :]
    sd_z = sd_z[0:x.shape[0], :]
    return mu_z, sd_z


def main():
    x_train, y_train = load_mnist_data('training')
    x_test, y_test = load_mnist_data('testing')
    label_train = np.argmax(y_train, 1)
    label_test = np.argmax(y_test, 1)

    model = SparseVae(784, 50, [200, 200], [200, 200], tf.nn.tanh, 0.0001)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graph', sess.graph)
        sess.run(tf.global_variables_initializer())

        train_model(model, sess, writer, x_train, 20, 100, 0.001)
        if not os.path.exists('model'):
            os.mkdir('model')
        saver.save(sess, 'model/model')

        mu_z, sd_z = test_model(model, sess, x_train)
        mu_z_label = []
        sd_z_label = []
        for digit in range(10):
            idx = [i for i, x in enumerate(label_train) if x == digit]
            for i in range(50):
                mu_z_label.append(mu_z[idx[i], :])
                sd_z_label.append(sd_z[idx[i], :])
        plt.imshow(mu_z_label)
        plt.show()
        


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
