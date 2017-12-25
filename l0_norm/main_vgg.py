from vgg import vgg
from dataset import make_cifar10_dataset, shuffle_data
import math


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


def main():
    x_train, y_train, x_test, y_test = make_cifar10_dataset()

    model = vgg('TRAIN')
    