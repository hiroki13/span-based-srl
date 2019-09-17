import theano
import theano.tensor as T

from nn.layers.core import Unit
from nn.utils import logsumexp


class SeqLabelAlg(Unit):
    def __init__(self, name='SeqLabelModel'):
        super(SeqLabelAlg, self).__init__(name=name)

    def viterbi(self, emit_scores, trans_scores):
        """
        :param emit_scores: 1D: n_words, 2D: batch_size, 3D: n_labels
        :param trans_scores: 1D: n_words, 2D: n_labels
        :return: 1D: n_words; 2D: batch_size, elem=label id
        """
        [scores, labels], _ = theano.scan(fn=self._viterbi_forward,
                                          sequences=[emit_scores[1:]],
                                          outputs_info=[emit_scores[0], None],
                                          non_sequences=trans_scores)

        label_max_last = T.argmax(scores[-1], axis=1)
        labels_max, _ = theano.scan(fn=self._viterbi_backward,
                                    sequences=labels[::-1],
                                    outputs_info=label_max_last)

        y = T.zeros(shape=(emit_scores.shape[0], emit_scores.shape[1]), dtype='int32')
        y = T.set_subtensor(y[-1], label_max_last)
        y = T.set_subtensor(y[:-1], labels_max[::-1])
        return y

    @staticmethod
    def _viterbi_forward(e_t, score_prev, trans):
        """
        :param e_t: 1D: batch_size, 2D: n_labels
        :param score_prev: 1D: batch_size, 2D: n_labels
        :param trans: 1D: n_labels, 2D, n_labels
        :return: max_scores_t: 1D: batch_size, 2D: n_labels
        :return: max_labels_t: 1D: batch_size, 2D: n_labels
        """
        score = score_prev.dimshuffle(0, 'x', 1) + trans + e_t.dimshuffle(0, 1, 'x')
        max_scores_t, max_labels_t = T.max_and_argmax(score, axis=2)
        return max_scores_t, max_labels_t

    @staticmethod
    def _viterbi_backward(labels_t, label_max):
        """
        :param labels_t: 1D: batch_size, 2D: n_labels; elem=label id
        :param label_max: 1D: batch_size; elem=label id
        :return: 1D: batch_size; elem=label id
        """
        return labels_t[T.arange(labels_t.shape[0]), label_max]


class CRF(SeqLabelAlg):
    def __init__(self,
                 input_dim,
                 output_dim,
                 use_bias=True,
                 weight_init='xavier',
                 bias_init='zero'):
        super(CRF, self).__init__(name='CRF(%dx%d)' % (input_dim, output_dim))
        self.W = self._set_param(shape=(input_dim, output_dim),
                                 init_type=weight_init,
                                 name='W_crf')
        self.W_t = self._set_param(shape=(output_dim, output_dim),
                                   init_type=weight_init,
                                   name='W_tran_crf')

        if use_bias:
            self.b = self._set_param(shape=output_dim,
                                     init_type=bias_init,
                                     name='b_crf')
            self.params = [self.W, self.W_t, self.b]
        else:
            self.b = None
            self.params = [self.W, self.W_t]

    def forward(self, x):
        emit_scores = T.dot(x, self.W)
        if self.b:
            emit_scores = emit_scores + self.b
        return emit_scores

    def get_y_proba(self, emit_scores, y_true):
        """
        :param emit_scores: 1D: n_words, 2D: batch_size, 3D: n_labels
        :param y_true: 1D: n_words, 2D: batch_size
        :return: 1D: batch_size; elem=log probability
        """
        # 1D: batch_size, 2D: n_labels
        z_score0 = emit_scores[0]
        # 1D: batch_size; elem=path score
        y_score0 = z_score0[T.arange(z_score0.shape[0]), y_true[0]]

        inputs = [emit_scores[1:], y_true[1:]]
        [_, y_scores, z_scores], _ = theano.scan(fn=self._forward_step,
                                                 sequences=inputs,
                                                 outputs_info=[y_true[0], y_score0, z_score0],
                                                 non_sequences=self.W_t)

        y_score = y_scores[-1]
        z_score = logsumexp(z_scores[-1], axis=1).flatten()

        return y_score - z_score

    @staticmethod
    def _forward_step(h_t, y_t, y_prev, y_score_prev, z_score_prev, trans):
        """
        :param h_t: 1D: batch_size, 2D: n_labels
        :param y_t: 1D: batch_size
        :param y_prev: 1D: batch_size
        :param y_score_prev: 1D: batch_size
        :param z_score_prev: 1D: batch_size, 2D: n_labels
        :param trans: 1D: n_labels, 2D, n_labels
        """
        # 1D: batch_size
        y_score_t = y_score_prev + trans[y_t, y_prev] + h_t[T.arange(h_t.shape[0]), y_t]
        # 1D: batch_size, 2D: n_labels, 3D: n_labels
        z_sum = z_score_prev.dimshuffle(0, 'x', 1) + trans
        # 1D: batch_size, 2D: n_labels
        z_score_t = logsumexp(z_sum, axis=2).reshape(h_t.shape) + h_t
        return y_t, y_score_t, z_score_t

    def get_y_pred(self, emit_scores):
        """
        :param emit_scores: 1D: n_words, 2D: batch_size, 3D: n_labels
        :return: 1D: batch_size, 2D: n_words; elem=label id
        """
        return self.viterbi(emit_scores=emit_scores, trans_scores=self.W_t).dimshuffle(1, 0)
