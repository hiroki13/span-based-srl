import theano.tensor as T


def categorical_accuracy(y_true, y_pred):
    return T.sum(T.eq(y_true, y_pred))


def log_likelihood(y_true, y_proba):
    y_true = y_true.flatten()
    y_proba = y_proba.reshape((y_proba.shape[0] * y_proba.shape[1], -1))
    return T.sum(T.log(y_proba[T.arange(y_true.shape[0]), y_true]))
