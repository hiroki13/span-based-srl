import sys
import time
import math
import glob

import numpy as np
import theano
import theano.tensor as T

from srl.models import SpanModel, MoEModel, CRFModel
from srl.decoders import Decoder
from nn.regularizers import L2Regularizer
from nn.optimizers import get_optimizer
from utils.evaluators import f_score, correct_and_pred_spans, metrics_for_bio
from utils.savers import save_pickle
from utils.loaders import load_pickle
from utils.misc import write


class SpanModelAPI(object):
    def __init__(self, argv):
        self.argv = argv

        self.model = None
        self.experts = None
        self.train_func = None
        self.pred_func = None

        self.vocab_word = None
        self.vocab_label = None
        self.vocab_label_valid = None

        self.input_dim = None
        self.hidden_dim = None
        self.output_dim = None
        self.use_elmo = None

        self.decoder = None
        self.optimizer = None

        self.n_true_spans = 0.

    def set_model(self, **kwargs):
        write('Setting a model...')
        argv = self.argv

        self.vocab_word = kwargs['vocab_word']
        self.use_elmo = kwargs['use_elmo']
        self.vocab_label = kwargs['vocab_label']
        self.vocab_label_valid = kwargs['vocab_label_valid']
        word_emb = kwargs['word_emb']
        vocab_word_size = self.vocab_word.size() if self.vocab_word else 0

        self.input_dim = argv.emb_dim if word_emb is None else word_emb.shape[1]
        self.hidden_dim = argv.hidden_dim
        self.output_dim = -1

        self.decoder = Decoder(argv=argv, vocab_label=self.vocab_label)

        self.model = SpanModel()
        self.model.compile(inputs=self._set_inputs(),
                           vocab_word_size=vocab_word_size,
                           use_elmo=self.use_elmo,
                           word_emb=word_emb,
                           input_dim=[self.input_dim, self.input_dim],
                           hidden_dim=self.hidden_dim,
                           feat_dim=2 * self.hidden_dim,
                           output_dim=self.vocab_label.size(),
                           n_layers=argv.n_layers,
                           drop_rate=argv.drop_rate)

        write('\t- {}'.format("\n\t- ".join([l.name for l in self.model.layers])))
        self._show_model_config()

    def set_ensemble_model(self, **kwargs):
        write('Setting a model...')
        argv = self.argv

        self.vocab_word = kwargs['vocab_word']
        self.use_elmo = kwargs['use_elmo']
        self.vocab_label = kwargs['vocab_label']
        self.vocab_label_valid = kwargs['vocab_label_valid']
        word_emb = kwargs['word_emb']
        vocab_word_size = self.vocab_word.size() if self.vocab_word else 0

        self.input_dim = argv.emb_dim if word_emb is None else word_emb.shape[1]
        self.hidden_dim = argv.hidden_dim
        self.output_dim = -1

        self.decoder = Decoder(argv=argv, vocab_label=self.vocab_label)

        #################
        # Set MoE model #
        #################
        inputs = self._set_inputs()
        self.model = MoEModel()
        self.model.compile(inputs=inputs,
                           feat_dim=2 * self.hidden_dim,
                           output_dim=self.vocab_label.size(),
                           drop_rate=argv.drop_rate,
                           n_experts=argv.n_experts)
        write('\t- {}\n'.format("\n\t- ".join([l.name for l in self.model.layers])))

        ###############
        # Set experts #
        ###############
        experts = []
        for _ in range(argv.n_experts):
            model = SpanModel()
            model.compile(inputs=self.model.inputs,
                          vocab_word_size=vocab_word_size,
                          use_elmo=self.use_elmo,
                          input_dim=[self.input_dim, self.input_dim],
                          hidden_dim=self.hidden_dim,
                          feat_dim=2 * self.hidden_dim,
                          output_dim=self.vocab_label.size(),
                          n_layers=argv.n_layers,
                          word_emb=word_emb,
                          drop_rate=argv.drop_rate)
            write('\t- {}\n'.format("\n\t- ".join([l.name for l in model.layers])))
            experts.append(model)

        self.experts = experts

    def _set_inputs(self):
        x = []
        if self.vocab_word:
            x.append(T.imatrix('x_word'))
        if self.use_elmo:
            x.append(T.ftensor4('x_elmo'))
        x.append(T.imatrix('x_mark'))
        assert len(x) > 1
        return x

    def _show_model_config(self):
        model = self.model
        write('Model configuration')
        write('\t- Input  Dim: {}'.format(self.input_dim))
        write('\t- Hidden Dim: {}'.format(self.hidden_dim))
        write('\t- Output Dim: {}'.format(self.output_dim))
        write('\t- Parameters: {}'.format(sum(len(x.get_value(borrow=True).ravel())
                                              for x in model.params)))

    def save_params(self, epoch=-1):
        argv = self.argv
        if argv.output_dir:
            dir_name = argv.output_dir
        else:
            dir_name = 'output'
        if argv.output_fn:
            file_name = '/param.%s.epoch-%d' % (argv.output_fn, epoch)
        else:
            file_name = '/param.epoch-%d' % epoch

        fn = dir_name + file_name
        params = [p.get_value(borrow=True) for p in self.model.params]
        save_pickle(fn=fn, data=params)

    def load_params(self, path):
        params = load_pickle(path)
        assert len(self.model.params) == len(params)
        for p1, p2 in zip(self.model.params, params):
            p1.set_value(p2)

    def load_experts_params(self, path):
        write('Loading experts params...')
        param_files = glob.glob(path + '/*')
        param_files = [fn for fn in param_files
                       if fn.split('/')[-1].startswith('param')]
        write("\t - Param Files: %s" % str(param_files))
        for i, path in enumerate(param_files[:self.argv.n_experts]):
            params = load_pickle(path)
            assert len(self.experts[i].params) == len(params)
            for p1, p2 in zip(self.experts[i].params, params):
                p1.set_value(p2)

    def set_init_ensemble_param(self):
        write('Initializing params...')
        W = np.zeros(shape=(2 * self.hidden_dim, self.vocab_label.size()),
                     dtype=theano.config.floatX)
        b = np.zeros(shape=self.vocab_label.size(),
                     dtype=theano.config.floatX)
        for model in self.experts:
            W += model.params[-2].get_value(borrow=True)
        for model in self.experts:
            b += model.params[-1].get_value(borrow=True)
        W = W / len(self.experts)
        b = b / len(self.experts)
        self.model.params[-2].set_value(W)
        self.model.params[-1].set_value(b)

    def set_train_func(self):
        write('Building a training function...')

        self.optimizer = get_optimizer(self.argv)
        self.optimizer.set_params(self.model.params)
        if self.argv.load_opt_param:
            self.optimizer.load_params(self.argv.load_opt_param)

        # 1D: batch_size * n_spans, 2D: [batch index, label id, span index]
        span_true = T.imatrix('span_true')

        # 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
        h_span = self.model.span_feats(inputs=self.model.inputs)
        # 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        span_score = self.model.label_layer.logit_scores(h=h_span)
        # 1D: batch_size, 2D: n_labels; label id
        span_pred = self.model.argmax_span(span_score=span_score)

        nll = self.model.loss(span_score, span_true)
        l2_reg = L2Regularizer()
        objective = nll + l2_reg(alpha=self.argv.reg,
                                 params=self.model.params)

        grads = T.grad(cost=objective, wrt=self.model.params)
        updates = self.optimizer(grads=grads, params=self.model.params)

        self.train_func = theano.function(
            inputs=self.model.inputs + [span_true],
            outputs=[objective, span_pred],
            updates=updates,
            mode='FAST_RUN'
        )

    def set_pred_func(self):
        write('Building a predicting function...')
        if self.argv.search == 'argmax':
            self.set_pred_argmax_func()
        else:
            self.set_pred_score_func()

    def set_pred_argmax_func(self):
        # 1D: batch_size, 2D: n_spans, 3D: hidden_dim
        h_span = self.model.span_feats(inputs=self.model.inputs)
        # 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        logits = self.model.label_layer.logit_scores(h_span)
        # 1D: batch_size, 2D: n_labels; span index
        span_pred = self.model.argmax_span(logits)

        self.pred_func = theano.function(
            inputs=self.model.inputs,
            outputs=span_pred,
            mode='FAST_RUN'
        )

    def set_pred_score_func(self):
        # 1D: batch_size, 2D: n_spans, 3D: hidden_dim
        h_span = self.model.span_feats(inputs=self.model.inputs)
        # 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        logits = self.model.label_layer.logit_scores(h_span)
        # 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        span_score = self.model.exp_score(logits)

        self.pred_func = theano.function(
            inputs=self.model.inputs,
            outputs=span_score,
            mode='FAST_RUN'
        )

    def set_ensemble_train_func(self):
        write('Building an ensemble training function...')

        self.optimizer = get_optimizer(self.argv)
        self.optimizer.set_params(self.model.params)
        if self.argv.load_opt_param:
            self.optimizer.load_params(self.argv.load_opt_param)

        # 1D: batch_size * n_spans, 2D: [batch index, label id, span index]
        span_true = T.imatrix('span_true')

        # 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
        h_span = self.model.feat_layer.forward(self.model.inputs,
                                               self.experts)
        # 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        logits = self.model.feat_layer.logit_scores(h=h_span)
        # 1D: batch_size, 2D: n_labels; span index
        span_pred = self.model.argmax_span(logits)

        nll = self.model.loss(logits, span_true)
        l2_reg = L2Regularizer()
        objective = nll + l2_reg(alpha=self.argv.reg,
                                 params=self.model.params)

        grads = T.grad(cost=objective, wrt=self.model.params)
        updates = self.optimizer(grads=grads,
                                 params=self.model.params)

        self.train_func = theano.function(
            inputs=self.model.inputs + [span_true],
            outputs=[objective, span_pred],
            updates=updates,
            mode='FAST_RUN'
        )

    def set_ensemble_pred_func(self):
        write('Building an ensemble predicting function...')
        if self.argv.search == 'argmax':
            self.set_ensemble_pred_argmax_func()
        else:
            self.set_ensemble_pred_score_func()

    def set_ensemble_pred_argmax_func(self):
        # 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
        h_span = self.model.feat_layer.forward(self.model.inputs,
                                               self.experts)
        # 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        span_score = self.model.feat_layer.logit_scores(h=h_span)
        # 1D: batch_size, 2D: n_labels; span index
        span_pred = self.model.argmax_span(span_score=span_score)

        self.pred_func = theano.function(
            inputs=self.model.inputs,
            outputs=span_pred,
            mode='FAST_RUN'
        )

    def set_ensemble_pred_score_func(self):
        # 1D: batch_size, 2D: n_spans, 3D: 2 * hidden_dim
        h_span = self.model.feat_layer.forward(self.model.inputs,
                                               self.experts)
        # 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        logits = self.model.feat_layer.logit_scores(h=h_span)
        # 1D: batch_size, 2D: n_labels, 3D: n_spans; score
        span_score = self.model.exp_score(logits)

        self.pred_func = theano.function(
            inputs=self.model.inputs,
            outputs=span_score,
            mode='FAST_RUN'
        )

    def train(self, batches):
        start = time.time()
        n_batches = 0.
        loss_total = 0.
        p_total = 0.
        correct = 0.

        self.model.feat_layer.is_train.set_value(1)
        if self.experts:
            for model in self.experts:
                model.feat_layer.is_train.set_value(1)

        for inputs in batches:
            n_batches += 1

            if n_batches % 100 == 0:
                sys.stdout.write("%d " % n_batches)
                sys.stdout.flush()

            n_words = len(inputs[0][0])
            if n_words < 2 or 100 < n_words:
                continue

            loss, span_pred = self.train_func(*inputs)

            if math.isnan(loss):
                write('\n\nNAN: Index: %d\n' % n_batches)
                exit()

            loss_total += loss
            correct_i, p_total_i = correct_and_pred_spans(span_true=inputs[-1],
                                                          span_pred=span_pred,
                                                          marks=inputs[1])
            correct += correct_i
            p_total += p_total_i

        self.model.feat_layer.is_train.set_value(0)
        if self.experts:
            for model in self.experts:
                model.feat_layer.is_train.set_value(0)

        avg_loss = loss_total / n_batches
        p, r, f = f_score(correct, p_total, self.n_true_spans)

        write('\n\tTime: %f seconds' % (time.time() - start))
        write('\tAverage Negative Log Likelihood: %f(%f/%d)' % (avg_loss, loss_total, n_batches))
        write('\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
            f, p, int(correct), int(p_total), r, int(correct), int(self.n_true_spans)))

    def predict(self, batches):
        if self.argv.search == 'argmax':
            return self.predict_argmax(batches)
        else:
            return self.predict_greedy(batches)

    def predict_argmax(self, batches):
        """
        :param batches: 1D: n_sents, 2D: n_prds, 3D: n_feats, 4D: n_words; elem=(x_w, x_m)
        :return: y: 1D: n_sents, 2D: n_prds, 3D: n_spans, 3D: [label_id, pre_index, post_index]
        """
        start = time.time()
        y = []

        for index, inputs in enumerate(batches):
            if (index + 1) % 100 == 0:
                sys.stdout.write("%d " % (index + 1))
                sys.stdout.flush()

            if len(inputs) == 0:
                span_triples = []
            else:
                span_pred = self.pred_func(*inputs)
                span_triples = self.decoder.argmax_span_triples(span_indices=span_pred,
                                                                marks=inputs[-1])
            y.append(span_triples)

        write('\n\tTime: %f seconds' % (time.time() - start))
        return y

    def predict_greedy(self, batches):
        """
        :param batches: 1D: n_sents, 2D: n_prds, 3D: n_feats, 4D: n_words; elem=(x_w, x_m)
        :return: y: 1D: n_sents, 2D: n_prds, 3D: n_spans, 3D: [label_id, pre_index, post_index]
        """
        start = time.time()
        y = []

        for index, inputs in enumerate(batches):
            if (index + 1) % 100 == 0:
                sys.stdout.write("%d " % (index + 1))
                sys.stdout.flush()

            if len(inputs) == 0:
                span_triples = []
            else:
                scores = self.pred_func(*inputs)
                span_triples = self.decoder.greedy_span_triples(scores=scores,
                                                                marks=inputs[-1])
            y.append(span_triples)

        write('\n\tTime: %f seconds' % (time.time() - start))
        return y


class BIOModelAPI(SpanModelAPI):
    def set_model(self, **kwargs):
        write('Setting a model...')
        argv = self.argv

        self.vocab_word = kwargs['vocab_word']
        self.use_elmo = kwargs['use_elmo']
        self.vocab_label = kwargs['vocab_label']
        self.vocab_label_valid = kwargs['vocab_label_valid']
        word_emb = kwargs['word_emb']
        vocab_word_size = self.vocab_word.size() if self.vocab_word else 0

        self.input_dim = argv.emb_dim if word_emb is None else word_emb.shape[1]
        self.hidden_dim = argv.hidden_dim
        self.output_dim = self.vocab_label.size()

        self.model = CRFModel()
        self.model.compile(inputs=self._set_inputs(),
                           vocab_word_size=vocab_word_size,
                           use_elmo=self.use_elmo,
                           word_emb=word_emb,
                           input_dim=[self.input_dim, self.input_dim],
                           hidden_dim=self.hidden_dim,
                           output_dim=self.output_dim,
                           n_layers=argv.n_layers,
                           init_emb=word_emb,
                           drop_rate=argv.drop_rate)

        write('\t- {}'.format("\n\t- ".join([l.name for l in self.model.layers])))
        self._show_model_config()

    def set_train_func(self):
        write('Building a training function...')

        self.optimizer = get_optimizer(self.argv)
        self.optimizer.set_params(self.model.params)
        if self.argv.load_opt_param:
            write('\tLoading optimization params...')
            self.optimizer.load_params(self.argv.load_opt_param)

        y_true = T.imatrix('y')

        # 1D: batch_size, 2D: n_words, 3D: output_dim
        emit_scores = self.model.get_emit_scores()
        # 1D: batch_size, 2D: n_words; elem=label id
        y_pred = self.model.label_layer.get_y_pred(emit_scores)
        # 1D: batch_size; elem=log proba
        y_path_proba = self.model.label_layer.get_y_path_proba(emit_scores, y_true)

        l2_reg = L2Regularizer()
        cost = - T.mean(y_path_proba) + l2_reg(alpha=self.argv.reg,
                                               params=self.model.params)

        grads = T.grad(cost=cost, wrt=self.model.params)
        updates = self.optimizer(grads=grads, params=self.model.params)

        self.train_func = theano.function(
            inputs=self.model.inputs + [y_true],
            outputs=[cost, y_pred],
            updates=updates,
            on_unused_input='warn',
            mode='FAST_RUN'
        )

    def set_pred_func(self):
        write('Building a predicting function...')

        # 1D: batch_size, 2D: n_words, 3D: output_dim
        o = self.model.get_emit_scores()
        # 1D: batch_size, 2D: n_words; elem=label id
        y_pred = self.model.label_layer.get_y_pred(o)

        self.pred_func = theano.function(
            inputs=self.model.inputs,
            outputs=y_pred,
            on_unused_input='warn',
            mode='FAST_RUN'
        )

    def train(self, batches):
        start = time.time()
        n_batches = 0.
        n_samples = 0.
        loss_total = 0.
        p_total = 0.
        r_total = 0.
        correct = 0.

        self.model.feat_layer.is_train.set_value(1)

        for index, inputs in enumerate(batches):
            if (index + 1) % 100 == 0:
                sys.stdout.write('%d ' % (index + 1))
                sys.stdout.flush()

            batch_size = len(inputs[0])
            n_words = len(inputs[0][0])
            if n_words < 2 or 100 < n_words:
                continue

            loss, y_pred = self.train_func(*inputs)

            if math.isnan(loss):
                write('\n\nNAN: Index: %d\n' % (index + 1))
                exit()

            loss_total += loss
            n_batches += 1
            n_samples += batch_size * n_words

            correct_i, p_total_i, r_total_i = metrics_for_bio(y_true=inputs[-1],
                                                              y_pred=y_pred,
                                                              vocab_label=self.vocab_label)
            correct += correct_i
            p_total += p_total_i
            r_total += r_total_i

        self.model.feat_layer.is_train.set_value(0)

        avg_loss = loss_total / n_batches
        p, r, f = f_score(correct, p_total, r_total)

        write('\n\tTime: %f seconds' % (time.time() - start))
        write('\tAverage Negative Log Likelihood: %f(%f/%d)' % (avg_loss, loss_total, n_batches))
        write('\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
            f, p, int(correct), int(p_total), r, int(correct), int(r_total)))

    def predict(self, batches):
        """
        :param batches: 1D: n_batches, 2D: n_words; elem=(x_w, x_m)
        :return: y: 1D: n_batches, 2D: batch_size; elem=(y_pred(1D:n_words), y_proba(float))
        """
        start = time.time()
        y = []

        for index, inputs in enumerate(batches):
            if (index + 1) % 100 == 0:
                sys.stdout.write("%d " % (index + 1))
                sys.stdout.flush()

            if len(inputs) == 0:
                y_pred = []
            elif len(inputs[0][0]) < 2:
                y_pred = [[0] for _ in range(len(inputs[0]))]
            else:
                y_pred = self.pred_func(*inputs)
            y.append(y_pred)

        write('\n\tTime: %f seconds' % (time.time() - start))
        return y
