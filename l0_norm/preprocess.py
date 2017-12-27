import numpy as np

def rgb2yuv(x):
    shape = x.shape
    assert len(shape) == 4
    assert shape[-1] == 3
    y = np.zeros(shape)
    y[:, :, :, 0] = 0.299 * x[:, :, :, 0] + 0.587 * x[:, :, :, 1] + 0.114 * x[:, :, :, 2]
    y[:, :, :, 1] = 0.492 * (x[:, :, :, 2] - y[:, :, :, 0])
    y[:, :, :, 2] = 0.877 * (x[:, :, :, 0] - y[:, :, :, 0])
    return y


def normalize(x, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(x, (0, 1, 2))
        std = np.std(x, (0, 1, 2))
        y = (x - mean) / std
        return y, mean, std
    else:
        assert x.shape[-1] == len(mean)
        assert x.shape[-1] == len(std)
        assert x.shape[-1] == mean.size 
        assert x.shape[-1] == std.size 
        y = (x - mean) / std
        return y


def preprocess(x, mean=None, std=None):
    y = rgb2yuv(x)
    y = normalize(y, mean, std)
    return y 