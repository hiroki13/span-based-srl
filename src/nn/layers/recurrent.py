import theano
import theano.tensor as T

from nn.layers.core import Unit, sigmoid, tanh


class LSTM(Unit):
    def __init__(self,
                 input_dim,
                 output_dim,
                 use_bias=True,
                 recurrent_init='orth',
                 bias_init='zero'):
        super(LSTM, self).__init__(name='LSTM(%dx%d)' % (input_dim, output_dim))

        self.input_dim = input_dim
        self.output_dim = output_dim

        # inout gate parameters
        self.W_xi = self._set_param(shape=(input_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_xi')
        self.W_hi = self._set_param(shape=(output_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_hi')
        self.W_ci = self._set_param(shape=output_dim,
                                    init_type='xavier',
                                    name='W_ci')

        # forget gate parameters
        self.W_xf = self._set_param(shape=(input_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_xf')
        self.W_hf = self._set_param(shape=(output_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_hf')
        self.W_cf = self._set_param(shape=output_dim,
                                    init_type='xavier',
                                    name='W_cf')

        # cell parameters
        self.W_xc = self._set_param(shape=(input_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_xc')
        self.W_hc = self._set_param(shape=(output_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_hc')

        # output gate parameters
        self.W_xo = self._set_param(shape=(input_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_xf')
        self.W_ho = self._set_param(shape=(output_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_hf')
        self.W_co = self._set_param(shape=output_dim,
                                    init_type='xavier',
                                    name='W_cf')

        if use_bias:
            self.b_xi = self._set_param(shape=output_dim,
                                        init_type=bias_init,
                                        name='b_xi')
            self.b_xf = self._set_param(shape=output_dim,
                                        init_type='one',
                                        name='b_xf')
            self.b_xc = self._set_param(shape=output_dim,
                                        init_type=bias_init,
                                        name='b_xc')
            self.b_xo = self._set_param(shape=output_dim,
                                        init_type=bias_init,
                                        name='b_xo')
            self.params = [self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf,
                           self.W_xc, self.W_hc, self.W_xo, self.W_ho, self.W_co,
                           self.b_xi, self.b_xf, self.b_xc, self.b_xo]
        else:
            self.b_xi = None
            self.b_xf = None
            self.b_xc = None
            self.b_xo = None
            self.params = [self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf,
                           self.W_xc, self.W_hc, self.W_xo, self.W_ho, self.W_co]

    def _step(self, xi_t, xf_t, xc_t, xo_t, h_tm1, c_tm1):
        i_t = sigmoid(xi_t + T.dot(h_tm1, self.W_hi) + c_tm1 * self.W_ci)
        f_t = sigmoid(xf_t + T.dot(h_tm1, self.W_hf) + c_tm1 * self.W_cf)
        c_t = f_t * c_tm1 + i_t * tanh(xc_t + T.dot(h_tm1, self.W_hc))
        o_t = sigmoid(xo_t + T.dot(h_tm1, self.W_ho) + c_t * self.W_co)
        h_t = o_t * tanh(c_t)
        return h_t, c_t

    def forward(self, x, h0=None, mask=None):
        xi = T.dot(x, self.W_xi) + self.b_xi
        xf = T.dot(x, self.W_xf) + self.b_xf
        xc = T.dot(x, self.W_xc) + self.b_xc
        xo = T.dot(x, self.W_xo) + self.b_xo

        inputs = [xi, xf, xc, xo]

        if h0 is None:
            h0 = T.zeros(shape=(x[0].shape[0], self.output_dim), dtype=theano.config.floatX)
        c0 = T.zeros(shape=(x[0].shape[0], self.output_dim), dtype=theano.config.floatX)

        [h, _], _ = theano.scan(fn=self._step,
                                sequences=inputs,
                                outputs_info=[h0, c0])
        return h
