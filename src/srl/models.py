import theano
import theano.tensor as T

from nn.layers.embeddings import Embedding, ElmoLayer
from nn.layers.core import Dense, Dropout
from nn.layers.stack import AlterBiRNNLayer
from nn.layers.seqlabel import CRF
from nn.utils import logsumexp3d


class Model(object):
    def __init__(self):
        self.is_train = theano.shared(0, borrow=True)
        self.inputs = None
        self.outputs = None
        self.dropout = None
        self.input_layers = []
        self.hidden_layers = []
        self.output_layers = []
        self.layers = []
        self.params = []

    def compile(self, **kwargs):
        raise NotImplementedError

    def _set_params(self):
        for l in self.layers:
            self.params += l.params


class FeatureLayer(Model):
    def compile(self, **kwargs):
        self._set_layers(kwargs)
        self._set_params()

    def calc_hidden_units(self, inputs):
        embs = []
        for i in range(len(inputs)):
            # 1D: batch_size, 2D: n_words, 3D: input_dim
            emb_i = self.input_layers[i].forward(x=inputs[i],
                                                 is_train=self.is_train)
            embs.append(emb_i)

        # 1D: batch_size, 2D: n_words, 3D: input_dim
        x = T.concatenate(tensor_list=embs, axis=2)
        # 1D: n_words, 2D: batch_size, 3D: hidden_dim
        h = self.hidden_layers[0].forward(x=x.dimshuffle(1, 0, 2),
                                          is_train=self.is_train)
        return h

    def _set_layers(self, args):
        x_w_dim, x_m_dim = args['input_dim']
        hidden_dim = args['hidden_dim']
        drop_rate = args['drop_rate']

        ################
        # Input layers #
        ################
        if args['vocab_word_size'] > 0:
            emb_word = Embedding(input_dim=args['vocab_word_size'],
                                 output_dim=x_w_dim,
                                 init_emb=args['word_emb'],
                                 param_fix=True,
                                 drop_rate=drop_rate,
                                 name='EmbWord')
            self.input_layers.append(emb_word)

        if args['use_elmo']:
            emb_elmo = ElmoLayer(drop_rate=0.5,
                                 name='EmbElmo')
            self.input_layers.append(emb_elmo)

        emb_mark = Embedding(input_dim=2,
                             output_dim=x_m_dim,
                             init_emb=None,
                             param_init='xavier',
                             param_fix=False,
                             drop_rate=drop_rate,
                             name='EmbMark')
        self.input_layers.append(emb_mark)

        #################
        # Hidden layers #
        #################
        if args['use_elmo']:
            hidden_input_dim = (len(self.input_layers) - 2) * x_w_dim + x_m_dim + 1024
        else:
            hidden_input_dim = (len(self.input_layers) - 1) * x_w_dim + x_m_dim
        hidden_layer = AlterBiRNNLayer(input_dim=hidden_input_dim,
                                       output_dim=hidden_dim,
                                       n_layers=args['n_layers'],
                                       unit_type=args['rnn_unit'],
                                       connect_type='dense',
                                       drop_rate=drop_rate)
        self.hidden_layers = [hidden_layer]
        self.layers = self.input_layers + self.hidden_layers


class SoftmaxLayer(Model):
    def compile(self, **kwargs):
        self._set_layers(kwargs)
        self._set_params()

    def _set_layers(self, args):
        layer = Dense(input_dim=args['feat_dim'],
                      output_dim=args['output_dim'],
                      activation='softmax')
        self.layers = [layer]

    def forward(self, h):
        """
        :param h: 1D: n_words, 2D: batch_size, 3D: hidden_dim
        :return: 1D: batch_size, 2D: n_words, 3D: output_dim; elem=proba
        """
        return self.layers[0].forward(x=h.dimshuffle(1, 0, 2))

    def get_y_pred(self, o):
        """
        :param o: 1D: batch_size, 2D: n_words, 3D: output_dim; elem=proba
        :return: 1D: batch_size, 2D: n_words; elem=label id
        """
        return T.argmax(o, axis=2)

    def get_y_path_proba(self, o, y_true):
        """
        :param o: 1D: batch_size, 2D: n_words, 3D: output_dim; elem=proba
        :param y_true: 1D: batch_size, 2D: n_words; elem=label id
        :return: 1D: batch_size; elem=log proba
        """
        o = o.reshape((o.shape[0] * o.shape[1], -1))
        y_proba = o[T.arange(o.shape[0]), y_true.flatten()].reshape(y_true.shape)
        return T.sum(T.log(y_proba), axis=1)


class CRFLayer(Model):
    def compile(self, **kwargs):
        self._set_layers(kwargs)
        self._set_params()

    def _set_layers(self, args):
        layer = CRF(input_dim=args['feat_dim'],
                    output_dim=args['output_dim'])
        self.layers = [layer]

    def forward(self, h):
        """
        :param h: 1D: n_words, 2D: batch_size, 3D: hidden_dim
        :return: 1D: batch_size, 2D: n_words, 3D: output_dim; elem=emit score
        """
        return self.layers[0].forward(x=h).dimshuffle(1, 0, 2)

    def get_y_pred(self, o):
        """
        :param o: 1D: batch_size, 2D: n_words, 3D: output_dim; elem=emit score
        :return: 1D: batch_size, 2D: n_words; elem=label id
        """
        return self.layers[0].get_y_pred(emit_scores=o.dimshuffle(1, 0, 2))

    def get_y_path_proba(self, o, y_true):
        """
        :param o: 1D: batch_size, 2D: n_words, 3D: output_dim; elem=emit score
        :param y_true: 1D: batch_size, 2D: n_words; elem=label id
        :return: 1D: batch_size; elem=log proba
        """
        return self.layers[0].get_y_proba(emit_scores=o.dimshuffle(1, 0, 2),
                                          y_true=y_true.dimshuffle(1, 0))


class LabelLayer(Model):
    def compile(self, **kwargs):
        self._set_layers(hidden_dim=kwargs['feat_dim'],
                         output_dim=kwargs['output_dim'])
        self._set_params()

    def _set_layers(self, hidden_dim, output_dim):
        self.layers = [Dense(input_dim=hidden_dim,
                             output_dim=output_dim)]

    def calc_feats(self, h):
        """
        :param h: 1D: n_words, 2D: batch_size, 3D: hidden_dim
        :return: 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
        """
        h = h.dimshuffle(1, 0, 2)
        n_words = h.shape[1]

        m = T.triu(T.ones(shape=(n_words, n_words)))
        indices = m.nonzero()

        # 1D: batch_size, 2D: n_spans, 3D: hidden_dim
        h_i = h[:, indices[0]]
        h_j = h[:, indices[1]]

        h_diff = h_i - h_j
        h_add = h_i + h_j

        return T.concatenate([h_add, h_diff], axis=2)

    def calc_logit_scores(self, h):
        """
        :param h: 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
        :return: 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        """
        return self.layers[-1].forward(h).dimshuffle(0, 2, 1)


class MoELabelLayer(LabelLayer):
    def __init__(self):
        super(MoELabelLayer, self).__init__()
        self.hidden_dim = -1

    def compile(self, **kwargs):
        self.dropout = Dropout(rate=kwargs['drop_rate'])
        self._set_layers(n_experts=kwargs['n_experts'],
                         hidden_dim=kwargs['feat_dim'],
                         output_dim=kwargs['output_dim'])

    def _set_layers(self, n_experts, hidden_dim, output_dim):
        mixture = Dense(input_dim=1,
                        output_dim=n_experts,
                        activation=None,
                        use_bias=False,
                        weight_init='zero',
                        bias_init='zero')
        hidden_layer = Dense(input_dim=hidden_dim,
                             output_dim=hidden_dim,
                             weight_init="orthone")
        output_layer = Dense(input_dim=hidden_dim,
                             output_dim=output_dim)
        self.hidden_dim = hidden_dim
        self.layers = [mixture, hidden_layer, output_layer]

    def forward(self, x, experts):
        """
        :param x: 1D: n_inputs, 2D: batch_size, 3D: n_words; feat id
        :param experts: 1D: n_experts; model
        :return: 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
        """
        # 1D: 1, 2D: n_experts, 3D: 1
        mixture = T.nnet.softmax(self.layers[0].W).dimshuffle('x', 1, 0)
        # 1D: 1, 2D: n_experts, 3D: 2 * hidden_dim
        mixture = T.repeat(mixture, repeats=self.hidden_dim, axis=2)

        batch_size = x[0].shape[0]
        n_words = x[0].shape[1]
        n_spans = T.cast(n_words * (n_words + 1) / 2, dtype='int32')

        # 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim; score
        h_span = T.zeros(shape=(batch_size, n_spans, self.hidden_dim),
                         dtype=theano.config.floatX)

        for i, expert in enumerate(experts):
            # 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
            h_span_tm = expert.calc_span_feats(inputs=x)
            h_span = h_span + mixture[:, i] * h_span_tm

        return self.layers[1].forward(h_span)


class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.feat_layer = None
        self.label_layer = None

    def compile(self, inputs, **kwargs):
        self.inputs = inputs
        self.feat_layer = FeatureLayer()
        self.feat_layer.compile(**kwargs)
        self.label_layer = SoftmaxLayer() if kwargs['seq_label_alg'] == 'softmax' else CRFLayer()
        self.label_layer.compile(**kwargs)
        self.layers = self.feat_layer.layers + self.label_layer.layers
        self._set_params()

    def get_output(self):
        """
        :return: 1D: batch_size, 2D: n_words, 3D: output_dim
        """
        h = self.feat_layer.calc_hidden_units(self.inputs)
        return self.label_layer.forward(h)


class SpanModel(BaseModel):
    def compile(self, inputs, **kwargs):
        self.inputs = inputs
        self.feat_layer = FeatureLayer()
        self.feat_layer.compile(**kwargs)
        self.label_layer = LabelLayer()
        self.label_layer.compile(**kwargs)
        self.layers = self.feat_layer.layers + self.label_layer.layers
        self._set_params()

    def calc_span_feats(self, inputs):
        """
        :param inputs: 1D: n_inputs, 2D: batch_size, 3D: n_words; feat id
        :return: 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
        """
        # 1D: n_words, 2D: batch_size, 3D: 2 * hidden_dim
        h_rnn = self.feat_layer.calc_hidden_units(inputs)
        return self.label_layer.calc_feats(h_rnn)

    @staticmethod
    def argmax_span(span_score):
        """
        :param span_score: 1D: batch_size, 2D: n_labels, 3D: n_spans
        :return: 1D: batch_size, 2D: n_labels; span index
        """
        return T.argmax(span_score, axis=2)

    @staticmethod
    def calc_loss(span_score, span_true):
        """
        :param span_score: 1D: batch_size, 2D: n_labels, 3D: n_spans
        :param span_true: 1D: batch_size * n_spans; (batch index, label id, span index)
        """
        batch_size = span_score.shape[0]

        # 1D: batch_size * n_spans; index
        batch_index = span_true[:, 0]
        label_index = span_true[:, 1]
        span_index = span_true[:, 2]

        # 1D: batch_size * n_spans; score
        true_span_score = span_score[batch_index, label_index, span_index]

        # 1D: batch_size, 2D: n_labels; elem=score
        z = logsumexp3d(span_score, axis=2)
        # 1D: batch_size * n_spans; score
        z = z[batch_index, label_index]

        # 1D: batch_size * n_spans; score
        nll = true_span_score - z

        return - T.sum(nll) / batch_size

    @staticmethod
    def calc_scores(span_score):
        """
        :param span_score: 1D: batch_size, 2D: n_labels, 3D: n_spans; logit score
        :return: 1D: batch_size, 2D: n_labels, 3D: n_spans
        """
        return T.exp(span_score)


class MoEModel(SpanModel):
    def compile(self, inputs, **kwargs):
        self.inputs = inputs
        self.feat_layer = MoELabelLayer()
        self.feat_layer.compile(**kwargs)
        self.layers = self.feat_layer.layers
        self._set_params()

