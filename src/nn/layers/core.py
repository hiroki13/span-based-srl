import theano
import theano.tensor as T

from nn.initializers import Zero, One, Identity, Uniform, Normal, Xavier, Orthonormal
from nn.activations import sigmoid, tanh, relu, softmax


class Unit(object):
    def __init__(self, name='unit'):
        self.name = name

    @staticmethod
    def _set_param(shape, init_type=None, name=None):
        if init_type == 'zero':
            init = Zero()
        elif init_type == 'one':
            init = One()
        elif init_type == 'xavier':
            init = Xavier()
        elif init_type == 'orth':
            init = Orthonormal()
        elif init_type == 'identity':
            init = Identity()
        elif init_type == 'uniform':
            init = Uniform()
        else:
            init = Normal()
        return init(shape=shape, name=name)

    @staticmethod
    def _set_activation(activation_type):
        if activation_type == 'sigmoid':
            return sigmoid
        elif activation_type == 'tanh':
            return tanh
        elif activation_type == 'relu':
            return relu
        elif activation_type == 'softmax':
            return softmax
        return None


class Dense(Unit):
    def __init__(self,
                 input_dim,
                 output_dim,
                 activation=None,
                 use_bias=True,
                 weight_init='xavier',
                 bias_init='zero'):
        super(Dense, self).__init__(name='Dense(%dx%d,%s)' % (input_dim, output_dim, activation))

        self.W = self._set_param(shape=(input_dim, output_dim),
                                 init_type=weight_init,
                                 name='W_dense')
        if use_bias:
            self.b = self._set_param(shape=output_dim,
                                     init_type=bias_init,
                                     name='b_dense')
            self.params = [self.W, self.b]
        else:
            self.b = None
            self.params = [self.W]

        self.activation = self._set_activation(activation)

    def forward(self, x):
        h = T.dot(x, self.W)
        if self.b:
            h = h + self.b
        if self.activation:
            h = self.activation(h)
        return h


class Dropout(Unit):
    """
    Reference: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting]
    """
    def __init__(self, rate, seed=0):
        super(Dropout, self).__init__(name='Dropout(p={:>1.1})'.format(rate))
        self.rate = min(1., max(0., rate))
        self.srng = T.shared_randomstreams.RandomStreams(seed=seed)

    def forward(self, x, is_train):
        drop_mask = self.srng.binomial(size=x.shape, n=1, p=1 - self.rate, dtype=theano.config.floatX)
        return T.switch(T.eq(is_train, 1), x * drop_mask, x * (1 - self.rate))
