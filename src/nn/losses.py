import theano.tensor as T


def binary_cross_entropy(output, target):
    return T.nnet.binary_crossentropy(output=output, target=target)


def negative_log_likelihood(y_proba, y_true=None):
    """
    :param y_proba: 1D: batch_size, 2D: n_words, 3D: n_words; elem=word id
    :param y_true: 1D: batch_size, 2D: n_words; elem=word id
    """
    if y_true:
        y_true_flatten = y_true.flatten()
        y_proba = y_proba.reshape((y_proba.shape[0] * y_proba.shape[1], y_proba.shape[2]))
        nll = - T.sum(T.log(y_proba[T.arange(y_true_flatten.shape[0]), y_true_flatten]).reshape(y_true.shape), axis=1)
    else:
        nll = - y_proba
    return nll
