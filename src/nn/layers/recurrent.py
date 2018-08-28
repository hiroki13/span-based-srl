import theano
import theano.tensor as T

from nn.layers.core import Unit, sigmoid, tanh


class GRU(Unit):
    def __init__(self,
                 input_dim,
                 output_dim,
                 use_bias=True,
                 recurrent_init='orth',
                 bias_init='zero'):
        super(GRU, self).__init__(name='GRU(%dx%d)' % (input_dim, output_dim))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_xr = self._set_param(shape=(input_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_xr')
        self.W_hr = self._set_param(shape=(output_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_hr')
        self.W_xz = self._set_param(shape=(input_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_xz')
        self.W_hz = self._set_param(shape=(output_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_hz')
        self.W_xh = self._set_param(shape=(input_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_xh')
        self.W_hh = self._set_param(shape=(output_dim, output_dim),
                                    init_type=recurrent_init,
                                    name='W_hh')

        if use_bias:
            self.b_xr = self._set_param(shape=output_dim,
                                        init_type=bias_init,
                                        name='b_xr')
            self.b_xh = self._set_param(shape=output_dim,
                                        init_type=bias_init,
                                        name='b_xh')
            self.b_xz = self._set_param(shape=output_dim,
                                        init_type=bias_init,
                                        name='b_xz')
            self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh,
                           self.b_xr, self.b_xz, self.b_xh]
        else:
            self.b_xr = None
            self.b_xh = None
            self.b_xz = None
            self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]

    def _step(self, xr_t, xz_t, xh_t, h_tm1):
        r_t = sigmoid(xr_t + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(xz_t + T.dot(h_tm1, self.W_hz))
        h_hat_t = tanh(xh_t + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        return h_t

    def _step_masked(self, xr_t, xz_t, xh_t, mask_t, h_tm1):
        r_t = sigmoid(xr_t + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(xz_t + T.dot(h_tm1, self.W_hz)) * mask_t
        h_hat_t = tanh(xh_t + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        return h_t

    def forward(self, x, h0=None, mask=None):
        xr = T.dot(x, self.W_xr) + self.b_xr
        xz = T.dot(x, self.W_xz) + self.b_xz
        xh = T.dot(x, self.W_xh) + self.b_xh

        inputs = [xr, xz, xh]

        if h0 is None:
            h0 = T.zeros(shape=(x[0].shape[0], self.output_dim), dtype=theano.config.floatX)

        if mask is None:
            step = self._step
        else:
            step = self._step_masked
            inputs += [mask]

        h, _ = theano.scan(fn=step,
                           sequences=inputs,
                           outputs_info=[h0])
        return h


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
        """
        :param x_t: 1D: Batch, 2D: n_in
        :param h_tm1: 1D: Batch, 2D: n_h
        :param c_tm1: 1D: Batch, 2D; n_h
        :return: h_t: 1D: Batch, 2D: n_h
        :return: c_t: 1D: Batch, 2D: n_h
        """
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
