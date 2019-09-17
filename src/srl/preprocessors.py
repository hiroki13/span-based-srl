from collections import Counter
from copy import deepcopy

import numpy as np

from utils.vocab import Vocab, UNK
from utils.sent import Conll05Sent, Conll12Sent
from utils.misc import span_to_span_index, make_vocab_from_ids
from utils.savers import save_key_value_format
from utils.loaders import load_key_value_format


class Preprocessor(object):
    def __init__(self, argv):
        self.argv = argv
        self.data_type = argv.data_type

    @staticmethod
    def make_vocab_word(word_list):
        vocab_word = Vocab()
        vocab_word.add_word(UNK)
        for w in word_list:
            vocab_word.add_word(w)
        return vocab_word

    def make_and_save_vocab_label(self,
                                  sents,
                                  vocab_label_init=None,
                                  save=False,
                                  load=False):
        argv = self.argv

        if load and argv.load_label:
            label_key_value = load_key_value_format(argv.load_label)
            vocab_label = make_vocab_from_ids(label_key_value)
        else:
            vocab_label = self.make_vocab_label(sents=sents,
                                                vocab_label_init=vocab_label_init)
        if save:
            if argv.output_dir:
                dir_name = argv.output_dir
            else:
                dir_name = 'output'
            if argv.output_fn:
                file_name = '/label_ids.' + argv.output_fn
            else:
                file_name = '/label_ids'

            fn = dir_name + file_name
            values, keys = map(lambda x: x, zip(*enumerate(vocab_label.i2w)))
            save_key_value_format(fn=fn, keys=keys, values=values)

        return vocab_label

    def make_sents(self, corpus):
        """
        :param corpus: 1D: n_sents, 2D: n_words
        :return: 1D: n_sents
        """
        if len(corpus) == 0:
            return []

        if self.data_type == 'conll05':
            column = 6
            gen_sent = Conll05Sent
        else:
            column = 12
            gen_sent = Conll12Sent

        is_test = True if len(corpus[0][0]) < column else False
        return [gen_sent(sent, is_test) for sent in corpus]

    @staticmethod
    def split_x_and_y(batches, index=-1):
        """
        :param batches: 1D: n_batches, 2D: batch_size; elem=(x, m, y)
        :param index: split column index
        :return 1D: n_batches, 2D: batch_size; elem=(x, m)
        :return 1D: n_batches, 2D: batch_size; elem=y
        """
        x = []
        y = []
        for batch in batches:
            x.append(batch[:index])
            y.append(batch[index])
        return x, y

    def make_batches(self,
                     samples,
                     is_valid_data=False,
                     shuffle=True):
        """
        :param samples: 1D: n_samples, 2D: [x, m, y]
        :param is_valid_data: boolean
        :param shuffle: boolean
        :return 1D: n_batches, 2D: batch_size; elem=[x, m, y]
        """
        if shuffle:
            np.random.shuffle(samples)
            samples.sort(key=lambda sample: len(sample[0]))

        batches = []
        batch = []
        prev_n_words = len(samples[0][0])

        for sample in samples:
            n_words = len(sample[0])
            if len(batch) == self.argv.batch_size or prev_n_words != n_words:
                batches.append(self._make_one_batch(batch, is_valid_data))
                batch = []
                prev_n_words = n_words
            batch.append(sample)

        if batch:
            batches.append(self._make_one_batch(batch, is_valid_data))

        if shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    @staticmethod
    def _make_one_batch(batch, is_valid_data):
        raise NotImplementedError

    @staticmethod
    def make_batch_per_sent(sents):
        """
        :param sents: 1D: n_sents; Sent()
        :return 1D: n_sents, 2D: n_prds; elem=[x, m]
        """
        batches = []
        for sent in sents:
            x = []

            x_word_ids = sent.word_ids
            if x_word_ids is not None:
                x.append(x_word_ids)

            x_elmo_emb = sent.elmo_emb
            if x_elmo_emb is not None:
                x.append(x_elmo_emb)

            batch = list(map(lambda m: x + [m], sent.mark_ids))
            batches.append(list(map(lambda b: b, zip(*batch))))

        return batches

    @staticmethod
    def set_sent_config(sents, elmo_emb, vocab_word, vocab_label):
        raise NotImplementedError

    @staticmethod
    def make_samples(sents, is_valid_data=False):
        raise NotImplementedError

    def make_vocab_label(self,
                         sents,
                         vocab_label_init=None):
        raise NotImplementedError


class SpanPreprocessor(Preprocessor):
    def make_vocab_label(self,
                         sents,
                         vocab_label_init=None):
        if len(sents) == 0:
            return None

        if vocab_label_init:
            vocab_label = deepcopy(vocab_label_init)
        else:
            vocab_label = Vocab()
            if self.argv.data_type == 'conll05':
                core_labels = ["A0", "A1", "A2", "A3", "A4", "A5"]
            else:
                core_labels = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"]
            for label in core_labels:
                vocab_label.add_word(label)

        bio_labels = []
        for sent in sents:
            for props in sent.prd_bio_labels:
                bio_labels += props
        cnt = Counter(bio_labels)
        bio_labels = [(w, c) for w, c in cnt.most_common()]

        for label, count in bio_labels:
            if not label.endswith('-V') and len(label) > 1:
                vocab_label.add_word(label[2:])

        return vocab_label

    @staticmethod
    def set_sent_config(sents, elmo_emb, vocab_word, vocab_label):
        for index, sent in enumerate(sents):
            sent.set_mark_ids()
            if vocab_word:
                sent.set_word_ids(vocab_word)
            if elmo_emb:
                sent.set_elmo_emb(elmo_emb[str(index)])
            if vocab_label:
                sent.set_span_triples(vocab_label)
                sent.set_span_triples_with_null(vocab_label.size())
        return sents

    @staticmethod
    def make_samples(sents, is_valid_data=False):
        samples = []

        for sent in sents:
            x = []

            x_word_ids = sent.word_ids
            if x_word_ids is not None:
                x.append(x_word_ids)

            x_elmo_emb = sent.elmo_emb
            if x_elmo_emb is not None:
                x.append(x_elmo_emb)

            if is_valid_data:
                triples = sent.span_triples
            else:
                triples = sent.span_triples_with_null

            assert len(sent.mark_ids) == len(triples)
            for m, spans in zip(sent.mark_ids, triples):
                # spans: 1D: n_spans, 2D: (r, i, j)
                samples.append(x + [m, spans])

        return samples

    @staticmethod
    def _make_one_batch(batch, is_valid_data):
        if is_valid_data:
            return list(map(lambda b: b, zip(*batch)))

        b = []
        y = []
        n_words = len(batch[0][0])
        for b_index, sample in enumerate(batch):
            b.append(sample[:-1])
            y_tmp = []
            for (r, i, j) in sample[-1]:
                span_index = span_to_span_index(i, j, n_words)
                y_tmp.append([b_index, r, span_index])
            y += y_tmp

        x = list(map(lambda b_i: b_i, zip(*b)))

        return x + [y]


class BIOPreprocessor(Preprocessor):
    def make_vocab_label(self,
                         sents,
                         vocab_label_init=None):
        if len(sents) == 0:
            return None

        if vocab_label_init:
            vocab_label = deepcopy(vocab_label_init)
        else:
            vocab_label = Vocab()
            none_label = 'O'
            vocab_label.add_word(none_label)

        labels = []
        for sent in sents:
            if sent.has_prds:
                for prop in sent.prd_bio_labels:
                    labels += prop
        cnt = Counter(labels)
        labels = [(w, c) for w, c in cnt.most_common()]

        for label, count in labels:
            vocab_label.add_word(label)

        return vocab_label

    @staticmethod
    def set_sent_config(sents, elmo_emb, vocab_word, vocab_label):
        for index, sent in enumerate(sents):
            sent.set_mark_ids()
            if vocab_word:
                sent.set_word_ids(vocab_word)
            if elmo_emb:
                sent.set_elmo_emb(elmo_emb[str(index)])
            if vocab_label:
                sent.set_label_ids(vocab_label)
        return sents

    @staticmethod
    def make_samples(sents, is_valid_data=False):
        samples = []

        for sent in sents:
            x = []

            x_word_ids = sent.word_ids
            if x_word_ids is not None:
                x.append(x_word_ids)

            x_elmo_emb = sent.elmo_emb
            if x_elmo_emb is not None:
                x.append(x_elmo_emb)

            assert len(sent.mark_ids) == len(sent.bio_label_ids)
            for m, spans in zip(sent.mark_ids, sent.bio_label_ids):
                samples.append(x + [m, spans])

        return samples

    @staticmethod
    def _make_one_batch(batch, is_valid_data):
        return list(map(lambda b: b, zip(*batch)))
