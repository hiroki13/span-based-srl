import theano.tensor as T


def softmax(x):
    if x.ndim == 3:
        x_shape = x.shape
        x = x.reshape((x_shape[0] * x_shape[1], x_shape[2]))
        return T.nnet.softmax(x).reshape(x_shape)
    elif x.ndim == 4:
        x_shape = x.shape
        x = x.reshape((x_shape[0] * x_shape[1] * x_shape[2], x_shape[3]))
        return T.nnet.softmax(x).reshape(x_shape)
    return T.nnet.softmax(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def relu(x):
    return T.nnet.relu(x)
