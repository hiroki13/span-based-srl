import numpy as np

from utils.misc import array, str_to_id
from utils.vocab import HYPH, UNK


class Sent(object):
    def __init__(self, sent, is_test=True):
        self.words = self._make_words(sent=sent, is_test=is_test)

        self.forms = [word.form for word in self.words]
        self.strings = [word.string for word in self.words]
        self.marks = self._set_marks(self.words)
        self.props = self._set_props(self.words)

        self.prd_indices = self._set_prd_indices(self.marks)
        self.prd_forms = [self.forms[i] for i in self.prd_indices]
        self.prd_props = self._set_prd_props(self.props)
        self.has_prds = True if len(self.prd_indices) > 0 else False

        self.n_words = len(sent)
        self.n_prds = len(self.prd_indices)

        self.word_ids = None
        self.mark_ids = None
        self.elmo_emb = None
        self.span_triples = None
        self.span_triples_with_null = None

    def _make_words(self, sent, is_test=True):
        return [self._make_word(line, is_test) for line in sent]

    @staticmethod
    def _make_word(line, is_test=True):
        raise NotImplementedError

    def _set_marks(self, words):
        raise NotImplementedError

    def _set_props(self, words):
        props = [word.prop for word in words]
        props = [self._make_bio_labels(prop) for prop in map(lambda p: p, zip(*props))]
        return list(map(lambda p: p, zip(*props)))

    @staticmethod
    def _set_prd_indices(marks):
        return [i for i, mark in enumerate(marks) if mark != HYPH]

    @staticmethod
    def _set_prd_props(props):
        return list(map(lambda p: p, zip(*props)))

    def set_word_ids(self, vocab_word):
        self.word_ids = array([w for w in str_to_id(sent=self.forms,
                                                    vocab=vocab_word,
                                                    unk=UNK)])

    def set_mark_ids(self):
        mark_ids = [[0 for _ in range(self.n_words)] for _ in range(self.n_prds)]
        for i, prd_index in enumerate(self.prd_indices):
            mark_ids[i][prd_index] = 1
        self.mark_ids = array(mark_ids)

    def set_elmo_emb(self, elmo_emb):
        """
        :param elmo_emb: 1D: n_layers, 2D: n_words, 3D: dim
        """
        elmo_emb = np.asarray(elmo_emb)
        elmo_emb = elmo_emb.transpose((1, 0, 2))
        assert len(elmo_emb) == self.n_words
        self.elmo_emb = elmo_emb

    @staticmethod
    def _make_bio_labels(prop):
        """
        :param prop: 1D: n_words; elem=bracket label
        :return: 1D: n_words; elem=BIO label
        """
        labels = []
        prev = None
        for arg in prop:
            if arg.startswith('('):
                if arg.endswith(')'):
                    prev = arg.split("*")[0][1:]
                    label = 'B-' + prev
                    prev = None
                else:
                    prev = arg[1:-1]
                    label = 'B-' + prev
            else:
                if prev:
                    label = 'I-' + prev
                    if arg.endswith(')'):
                        prev = None
                else:
                    label = 'O'
            labels.append(label)
        return labels

    @staticmethod
    def _get_spans(props):
        """
        :param props: 1D: n_words; elem=bio-label
        :return: 1D: n_spans; elem=[label, i, j]
        """
        spans = []
        span = []
        for w_i, label in enumerate(props):
            if label[-2:] == '-V':
                continue
            if label.startswith('B-'):
                if span:
                    spans.append(span)
                span = [label[2:], w_i, w_i]
            elif label.startswith('I-'):
                if span:
                    if label[2:] == span[0]:
                        span[2] = w_i
                    else:
                        spans.append(span)
                        span = [label[2:], w_i, w_i]
                else:
                    span = [label[2:], w_i, w_i]
            else:
                if span:
                    spans.append(span)
                span = []
        if span:
            spans.append(span)
        return spans

    def set_span_triples(self, vocab_label):
        """
        :param vocab_label: Vocab (labels); e.g. A0, A1
        """
        triples = []
        for props in self.prd_props:
            triples_tmp = []
            for (label, i, j) in self._get_spans(props):
                r = vocab_label.get_id(label)
                triples_tmp.append((r, i, j))
            triples.append(triples_tmp)
        self.span_triples = triples

    def set_span_triples_with_null(self, n_labels):
        assert len(self.span_triples) == len(self.prd_indices)
        triples_with_null = []
        for prd_index, spans in zip(self.prd_indices, self.span_triples):
            used_labels = [r for (r, i, j) in spans]
            null_spans = [(r, prd_index, prd_index)
                          for r in range(n_labels)
                          if r not in used_labels]
            triples = spans + null_spans
            triples.sort(key=lambda s: s[0])
            triples_with_null.append(triples)
        self.span_triples_with_null = triples_with_null


class Conll05Sent(Sent):
    @staticmethod
    def _make_word(line, is_test=False):
        return Word(form=line[0],
                    mark=line[5] if is_test is False else line[4],
                    sense=line[4] if is_test is False else None,
                    prop=line[6:] if is_test is False else [])

    def _set_marks(self, words):
        return [word.mark for word in words]


class Conll12Sent(Sent):
    @staticmethod
    def _make_word(line, is_test=False):
        return Word(form=line[3],
                    mark=line[6],
                    sense=line[7],
                    prop=line[11:-1] if is_test is False else [])

    def _set_marks(self, words):
        return list(map(lambda w: w.mark if w.sense != HYPH else HYPH, words))


class Word(object):
    def __init__(self, form, mark, sense, prop):
        self.form = form.lower()
        self.string = form
        self.mark = mark
        self.sense = sense
        self.prop = prop
