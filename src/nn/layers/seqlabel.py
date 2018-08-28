import theano
import theano.tensor as T

from nn.layers.core import Unit
from nn.utils import logsumexp, log0


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

    def forward(self, x, mask=None):
        emit_scores = T.dot(x, self.W)
        if self.b:
            emit_scores = emit_scores + self.b
        if mask is None:
            return emit_scores
        return emit_scores * mask

    def get_y_proba(self, emit_scores, y_true, mask=None):
        """
        :param emit_scores: 1D: n_words, 2D: batch_size, 3D: n_labels
        :param y_true: 1D: n_words, 2D: batch_size
        :param mask: 1D: n_words, 2D: batch_size, 3D: 1; elem=0/1
        :return: 1D: batch_size; elem=log probability
        """
        # 1D: batch_size, 2D: n_labels
        z_score0 = emit_scores[0]
        # 1D: batch_size; elem=path score
        y_score0 = z_score0[T.arange(z_score0.shape[0]), y_true[0]]

        inputs = [emit_scores[1:], y_true[1:]]
        if mask is None:
            step = self._step
        else:
            step = self._step_masked
            inputs += [mask[1:]]

        [_, y_scores, z_scores], _ = theano.scan(fn=step,
                                                 sequences=inputs,
                                                 outputs_info=[y_true[0], y_score0, z_score0],
                                                 non_sequences=self.W_t)

        y_score = y_scores[-1]
        z_score = logsumexp(z_scores[-1], axis=1).flatten()

        return y_score - z_score

    def get_y_pred(self, emit_scores):
        """
        :param emit_scores: 1D: n_words, 2D: batch_size, 3D: n_labels
        :return: 1D: batch_size, 2D: n_words; elem=label id
        """
        return self.viterbi(emit_scores=emit_scores, trans_scores=self.W_t).dimshuffle(1, 0)

    @staticmethod
    def _step(h_t, y_t, y_prev, y_score_prev, z_score_prev, trans):
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

    @staticmethod
    def _step_masked(h_t, y_t, mask_t, y_prev, y_score_prev, z_score_prev, trans):
        """
        :param h_t: 1D: batch_size, 2D: n_labels
        :param y_t: 1D: batch_size
        :param mask_t: 1D: batch_size, 2D: 1
        :param y_prev: 1D: batch_size
        :param y_score_prev: 1D: batch_size
        :param z_score_prev: 1D: batch_size, 2D: n_labels
        :param trans: 1D: n_labels, 2D, n_labels
        """
        # 1D: batch_size
        y_score_tmp = y_score_prev + trans[y_t, y_prev] + h_t[T.arange(h_t.shape[0]), y_t]
        y_score_t = mask_t.flatten() * y_score_tmp + (1. - mask_t.flatten()) * y_score_prev

        # 1D: batch_size, 2D: n_labels, 3D: n_labels
        z_sum = z_score_prev.dimshuffle(0, 'x', 1) + trans
        # 1D: batch_size, 2D: n_labels
        z_score_tmp = logsumexp(z_sum, axis=2).reshape(h_t.shape) + h_t
        z_score_t = mask_t * z_score_tmp + (1. - mask_t) * z_score_prev

        return y_t, y_score_t, z_score_t


class ViterbiCoreArgConstraint(SeqLabelAlg):
    def __init__(self):
        super(ViterbiCoreArgConstraint, self).__init__(name='SemiMarkovCRF')

    @staticmethod
    def forward_step(j,
                             score_hat_prev,
                             score_matrix):
        """
        :param j: scalar
        :param score_hat_prev: 1D: batch_size, 2D: n_words + 1
        :param score_matrix: 1D: batch_size, 2D: n_words(i), 3D: n_words(j), 4D: n_labels; elem=exp score with masking
        """
        # 1D: batch_size, 2D: j, 3D: n_labels; elem=score
        #        score_j = log0(score_matrix[:, :j + 1, j])
        score_j = score_matrix[:, :j + 1, j]

        # 1D: batch_size, 2D: j, 3D: n_labels; elem=score
        score = score_hat_prev[:, :j + 1].dimshuffle(0, 1, 'x') + score_j

        # 1D: batch_size, 2D: j
        score_label_hat, label_hat = T.max_and_argmax(score, axis=2)

        # 1D: batch_size
        score_hat, i_hat = T.max_and_argmax(score_label_hat, axis=1)

        # 1D: batch_size
        label_hat = label_hat[T.arange(label_hat.shape[0]), i_hat]

        # 1D: batch_size, 2D: n_words + 1
        score_hat_prev = T.set_subtensor(score_hat_prev[:, j + 1], score_hat)

        return i_hat, label_hat, score_hat_prev

    @staticmethod
    def backward_step(i_hat,
                      j,
                      i_hat_prev):
        """
        :param i_hat: 1D: batch_size
        :param j: scalar
        :param i_hat_prev: 1D: batch_size
        """
        i = T.switch(T.gt(i_hat_prev, j), i_hat, -1)
        i_hat_prev = T.switch(T.gt(i_hat_prev, j), i_hat, i_hat_prev)
        return i, j - 1, i_hat_prev

    def viterbi(self, score_matrix):
        """
        :param score_matrix: 1D: batch_size, 2D: n_words(i), 3D: n_words(j), 4D: n_labels; elem=score
        :return: z_score: 1D: batch_size, 2D: n_words; elem=partition value
        :return: i: 1D: batch_size, 2D: n_words; elem=pre_index of each span
        :return: label_hat: 1D: batch_size, 2D: n_words; elem=label id
        """

        batch_size = score_matrix.shape[0]
        n_words = score_matrix.shape[1]

        # 1D: batch_size, 2D: n_words
        score0 = T.zeros(shape=(batch_size, n_words + 1),
                         dtype='float32')

        # Forward
        [i_hat, label_hat, score], _ = theano.scan(fn=self.forward_step,
                                                   sequences=[T.arange(n_words)],
                                                   outputs_info=[None, None, score0],
                                                   non_sequences=score_matrix)

        # Backward
        i_hat0 = T.ones(shape=(1, batch_size), dtype='int64') * n_words
        [i, _, _], _ = theano.scan(fn=self.backward_step,
                                   sequences=[i_hat[::-1]],
                                   outputs_info=[None, n_words - 1, i_hat0[0]])
        i = i[::-1].dimshuffle(1, 0)

        return i, label_hat.dimshuffle(1, 0)


class SemiMarkovCRF(SeqLabelAlg):
    def __init__(self):
        super(SemiMarkovCRF, self).__init__(name='SemiMarkovCRF')

    @staticmethod
    def forward_step(j,
                     score_hat_prev,
                     z_score,
                     score_matrix):
        """
        :param j: scalar
        :param score_hat_prev: 1D: batch_size, 2D: n_words + 1
        :param z_score: 1D: batch_size, 2D: n_words + 1
        :param score_matrix: 1D: batch_size, 2D: n_words(i), 3D: n_words(j), 4D: n_labels; elem=exp score with masking
        """
        # 1D: batch_size, 2D: j, 3D: n_labels; elem=score
        score_j = score_matrix[:, :j + 1, j]

        # 1D: batch_size, 2D: j, 3D: n_labels; elem=score
        score = score_hat_prev[:, :j + 1].dimshuffle(0, 1, 'x') + score_j

        # 1D: batch_size, 2D: j
        score_label_hat, label_hat = T.max_and_argmax(score, axis=2)

        # 1D: batch_size
        score_hat, i_hat = T.max_and_argmax(score_label_hat, axis=1)

        # 1D: batch_size
        label_hat = label_hat[T.arange(label_hat.shape[0]), i_hat]

        # 1D: batch_size, 2D: n_words + 1
        score_hat_prev = T.set_subtensor(score_hat_prev[:, j + 1], score_hat)

        # 1D: batch_size, 2D: j * n_labels; elem=score
        z_score_tm = z_score[:, :j + 1].dimshuffle(0, 1, 'x') + score_j
        z_score_tm = z_score_tm.reshape((z_score_tm.shape[0], -1))

        # 1D: batch_size; elem=score
        z_score = T.set_subtensor(z_score[:, j + 1],
                                  logsumexp(z_score_tm, axis=1, keepdim=False))

        return i_hat, label_hat, score_hat_prev, z_score

    @staticmethod
    def backward_step(i_hat,
                      j,
                      i_hat_prev):
        """
        :param i_hat: 1D: batch_size
        :param j: scalar
        :param i_hat_prev: 1D: batch_size
        """
        i = T.switch(T.gt(i_hat_prev, j), i_hat, -1)
        i_hat_prev = T.switch(T.gt(i_hat_prev, j), i_hat, i_hat_prev)
        return i, j - 1, i_hat_prev

    def forward_backward(self, score_matrix):
        """
        :param score_matrix: 1D: batch_size, 2D: n_words(i), 3D: n_words(j), 4D: n_labels; elem=score
        :return: z_score: 1D: batch_size; elem=partition value
        :return: i: 1D: batch_size, 2D: n_words; elem=pre_index of each span
        :return: label_hat: 1D: batch_size, 2D: n_words; elem=label id
        """

        batch_size = score_matrix.shape[0]
        n_words = score_matrix.shape[1]

        # 1D: batch_size, 2D: n_words
        score0 = T.zeros(shape=(batch_size, n_words + 1),
                         dtype='float32')
        # 1D: batch_size, 2D: n_words
        z_score0 = T.zeros(shape=(batch_size, n_words + 1),
                           dtype='float32')

        # Forward
        [i_hat, label_hat, score, z_score], _ = theano.scan(fn=self.forward_step,
                                                            sequences=[T.arange(n_words)],
                                                            outputs_info=[None, None, score0, z_score0],
                                                            non_sequences=score_matrix)

        # Backward
        i_hat0 = T.ones(shape=(1, batch_size), dtype='int64') * n_words
        [i, _, _], _ = theano.scan(fn=self.backward_step,
                                   sequences=[i_hat[::-1]],
                                   outputs_info=[None, n_words - 1, i_hat0[0]])
        i = i[::-1].dimshuffle(1, 0)

        return z_score[-1, :, -1], i, label_hat.dimshuffle(1, 0)

    @staticmethod
    def viterbi_forward_step(j,
                             score_hat_prev,
                             score_matrix):
        """
        :param j: scalar
        :param score_hat_prev: 1D: batch_size, 2D: n_words + 1
        :param score_matrix: 1D: batch_size, 2D: n_words(i), 3D: n_words(j), 4D: n_labels; elem=exp score with masking
        """
        # 1D: batch_size, 2D: j, 3D: n_labels; elem=score
        #        score_j = log0(score_matrix[:, :j + 1, j])
        score_j = score_matrix[:, :j + 1, j]

        # 1D: batch_size, 2D: j, 3D: n_labels; elem=score
        score = score_hat_prev[:, :j + 1].dimshuffle(0, 1, 'x') + score_j

        # 1D: batch_size, 2D: j
        score_label_hat, label_hat = T.max_and_argmax(score, axis=2)

        # 1D: batch_size
        score_hat, i_hat = T.max_and_argmax(score_label_hat, axis=1)

        # 1D: batch_size
        label_hat = label_hat[T.arange(label_hat.shape[0]), i_hat]

        # 1D: batch_size, 2D: n_words + 1
        score_hat_prev = T.set_subtensor(score_hat_prev[:, j + 1], score_hat)

        return i_hat, label_hat, score_hat_prev

    def viterbi(self, score_matrix):
        """
        :param score_matrix: 1D: batch_size, 2D: n_words(i), 3D: n_words(j), 4D: n_labels; elem=score
        :return: z_score: 1D: batch_size, 2D: n_words; elem=partition value
        :return: i: 1D: batch_size, 2D: n_words; elem=pre_index of each span
        :return: label_hat: 1D: batch_size, 2D: n_words; elem=label id
        """

        batch_size = score_matrix.shape[0]
        n_words = score_matrix.shape[1]

        # 1D: batch_size, 2D: n_words
        score0 = T.zeros(shape=(batch_size, n_words + 1),
                         dtype='float32')

        # Forward
        [i_hat, label_hat, score], _ = theano.scan(fn=self.viterbi_forward_step,
                                                   sequences=[T.arange(n_words)],
                                                   outputs_info=[None, None, score0],
                                                   non_sequences=score_matrix)

        # Backward
        i_hat0 = T.ones(shape=(1, batch_size), dtype='int64') * n_words
        [i, _, _], _ = theano.scan(fn=self.backward_step,
                                   sequences=[i_hat[::-1]],
                                   outputs_info=[None, n_words - 1, i_hat0[0]])
        i = i[::-1].dimshuffle(1, 0)

        return i, label_hat.dimshuffle(1, 0)
