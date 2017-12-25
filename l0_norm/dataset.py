import numpy as np
from mnist import MNIST
import os 
import pickle


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        x = datadict[b'data']
        y = datadict[b'labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        y = np.array(y)
        return x, y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        x, y = load_CIFAR_batch(f)
        xs.append(x)
        ys.append(y)
    xtr = np.concatenate(xs)
    ytr = np.concatenate(ys)
    del x, y
    xte, yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return xtr / 255.0, ytr, xte / 255.0, yte


def make_cifar10_dataset(vectorize=False):
    num_classes = 10
    num_train   = 50000
    num_test    = 10000

    cifar10_dir = 'D:/v-bindai/Projects/SparseNN/data/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)

    # reshape to vectors
    if vectorize:
        x_train = np.reshape(x_train,(x_train.shape[0],-1))
        x_test  = np.reshape(x_test,(x_test.shape[0],-1))

    # make one-hot coding
    y_train_temp = np.zeros((num_train,num_classes))
    for i in range(num_train):
        y_train_temp[i,y_train[i]] = 1
    y_train = y_train_temp

    y_test_temp = np.zeros((num_test,num_classes))
    for i in range(num_test):
        y_test_temp[i,y_test[i]] = 1
    y_test = y_test_temp

    return x_train, y_train, x_test, y_test


def shuffle_data(x, y=None):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx, :]
    if y.any():
        y = y[idx]
        return x, y
    else:
        return x


def load_mnist_data(flag='training'):
    mndata = MNIST('D:/v-bindai/Projects/SparseNN/data/MNIST')
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
        elif flag == 'testing':
            images, labels = mndata.load_testing()
        else:
            raise Exception('Flag should be either training or testing.')
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images) / 255
    labels_array = np.array(labels)
    one_hot_labels = np.zeros((labels_array.size, labels_array.max() + 1))
    one_hot_labels[np.arange(labels_array.size), labels_array] = 1
    return images_array, one_hot_labels