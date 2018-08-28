import numpy as np
import theano
import theano.tensor as T

from nn.layers.core import Unit, Dropout


class Embedding(Unit):
    def __init__(self,
                 input_dim,
                 output_dim,
                 init_emb=None,
                 param_init='xavier',
                 param_fix=False,
                 drop_rate=0.0,
                 name=None):
        super(Embedding, self).__init__(name=name if name else 'Emb(%dx%d)' % (input_dim, output_dim))
        self.dropout = Dropout(drop_rate)

        self.W = self._set_weight(input_dim, output_dim, init_emb, param_init)
        if param_fix:
            self.params = []
        else:
            self.params = [self.W]

    def _set_weight(self, input_dim, output_dim, init_emb, param_init):
        if init_emb is None:
            return self._set_param(shape=(input_dim, output_dim),
                                   init_type=param_init,
                                   name='embedding')
        return theano.shared(init_emb)

    def forward(self, x, is_train=0):
        return self.dropout.forward(x=self.W[x], is_train=is_train)


class ElmoLayer(Unit):
    def __init__(self, drop_rate=0.0, name=None):
        super(ElmoLayer, self).__init__(name=name if name else 'ElmoEmb')
        self.dropout = Dropout(drop_rate)

        self.gamma = theano.shared(value=np.asarray([[1.0]], theano.config.floatX),
                                   name='gamma',
                                   borrow=True)
        self.scalar_mix = theano.shared(value=np.zeros(shape=(1, 3), dtype=theano.config.floatX),
                                        name='scalar_mix',
                                        borrow=True)
        self.params = [self.gamma, self.scalar_mix]

    def forward(self, x, is_train=0):
        """
        :param x: 1D: batch_size, 2D: n_words, 3D: n_layers, 4D: dim
        :param is_train: 0/1
        :return:
        """
        s = T.nnet.softmax(self.scalar_mix).dimshuffle('x', 'x', 1, 0)
        s = T.repeat(s, repeats=x.shape[3], axis=3)
        x = self.gamma[0, 0] * T.sum(s * x, axis=2)
        return self.dropout.forward(x=x, is_train=is_train)
