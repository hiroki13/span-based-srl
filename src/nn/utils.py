import numpy as np
import theano.tensor as T


def normalize_3d(x, eps=1e-8):
    l2 = x.norm(2, axis=2).dimshuffle((0, 1, 'x'))
    return x / (l2 + eps)


def logsumexp(x, axis, keepdim=True):
    """
    :param x: 1D: batch, 2D: n_y, 3D: n_y
    :return: 1D: batch, 2D: n_y, 3D: n_y
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    if keepdim:
        return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=keepdim)) + x_max
    return T.log(T.sum(T.exp(x - x_max), axis=axis)) + x_max.dimshuffle(0)


def logsumexp3d(x, axis=2):
    # 1D: batch_size, 2D: n_labels, 3D: 1
    x_max = T.max(x, axis=axis, keepdims=True)
    # 1D: batch_size, 2D: n_labels
    return T.log(T.sum(T.exp(x - x_max), axis=axis)) + x_max.dimshuffle(0, 1)


def log0(x):
    return T.switch(T.eq(x, 0.0), 0.0, T.log(x))


def frobenius_norm(matrix):
    if type(matrix) is list:
        return T.sqrt(T.sum(map(lambda m: T.sum(m ** 2), matrix)))
    return T.sqrt(T.maximum(T.sum(T.sqr(matrix)), 1e-8))


def np_frobenius_norm(matrix):
    return np.sqrt(np.sum(matrix**2))


def layer_normalization(x, axis=1, eps=1e-8):
    return (x - x.mean(axis=axis, keepdims=True)) / T.sqrt((x.var(axis=axis, keepdims=True) + eps))
