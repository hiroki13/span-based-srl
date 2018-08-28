import theano.tensor as T

from nn.layers.core import Dense, Dropout
from nn.layers.recurrent import LSTM


class StackLayer(object):
    def __init__(self, name='StackLayer'):
        self.name = name
        self.layers = []
        self.params = []

    def _set_layers(self):
        raise NotImplementedError

    @staticmethod
    def _set_rnn_unit(unit_type):
        return LSTM

    @staticmethod
    def _set_connect_unit(connect_type):
        return Dense

    def _set_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params)
        return params

    def forward(self, x, **kwargs):
        raise NotImplementedError


class BiRNNLayer(StackLayer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_layers,
                 unit_type,
                 connect_type,
                 drop_rate=0.0):
        name = 'BiRNNs-%d:(%dx%d)' % (n_layers, input_dim, output_dim)
        super(BiRNNLayer, self).__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.rnn_unit = self._set_rnn_unit(unit_type)
        self.connect_unit = self._set_connect_unit(connect_type)
        self.dropout = Dropout(drop_rate)

        self.layers = self._set_layers()
        self.params = self._set_params()

    def _set_layers(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0:
                rnn_input_dim = self.input_dim
                connect_input_dim = self.input_dim + self.output_dim
            else:
                rnn_input_dim = self.output_dim
                connect_input_dim = self.output_dim * 2

            r_unit = self.rnn_unit(input_dim=rnn_input_dim,
                                   output_dim=self.output_dim)
            c_unit = self.connect_unit(input_dim=connect_input_dim,
                                       output_dim=self.output_dim,
                                       activation='relu')
            layers += [r_unit, c_unit]
        return layers

    def forward(self, x, mask=None, is_train=False):
        n_layers = int(len(self.layers) / 2)
        for i in range(n_layers):
            if mask is None:
                h = self.layers[i * 2].forward(x=x)
                h = self.dropout.forward(x=h, is_train=is_train)
                x = self.layers[i * 2 + 1].forward(T.concatenate([x, h], axis=2))
            else:
                h = self.layers[i * 2].forward(x=x, mask=mask)
                h = self.dropout.forward(x=h, is_train=is_train)
                x = self.layers[i * 2 + 1].forward(T.concatenate([x, h], axis=2)) * mask
                mask = mask[::-1]
            x = x[::-1]
        if (n_layers % 2) == 1:
            return x[::-1]
        return x
