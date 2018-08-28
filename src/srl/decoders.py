from copy import deepcopy
from itertools import combinations_with_replacement as comb

from utils.misc import convert_span_to_span_index


class Decoder(object):
    def __init__(self, argv, vocab_label):
        self.argv = argv
        self.core_label_ids = self.set_core_labels(vocab_label)
        self.span_list = None

    def set_core_labels(self, vocab_label):
        if self.argv.data_type == 'conll05':
            core_labels = ["A0", "A1", "A2", "A3", "A4", "A5"]
        else:
            core_labels = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"]
        return [vocab_label.get_id(label)
                for label in core_labels
                if vocab_label.has_key(label)]

    def argmax_span_triples(self, span_indices, marks):
        """
        :param span_indices: 1D: batch_size, 2D; n_labels; span index
        :param marks: 1D: batch_size, 2D; n_words
        :return: 1D: batch_size, 2D: n_spans; [r, i, j]
        """
        n_words = len(marks[0])
        self.span_list = list(comb(range(n_words), 2))
        return [self._argmax_search(span_indices_i, mark)
                for span_indices_i, mark in zip(span_indices, marks)]

    def _argmax_search(self, span_indices, mark):
        spans = []
        prd_index = mark.nonzero()[0][0]
        for r, span_index in enumerate(span_indices):
            (i, j) = self.span_list[span_index]
            if i <= prd_index <= j:
                continue
            spans.append([r, i, j])
        return spans

    def greedy_span_triples(self, scores, marks):
        """
        :param scores: 1D: batch_size, 2D; n_labels, 3D: n_spans; score
        :param marks: 1D: batch_size, 2D; n_words
        :return: 1D: batch_size, 2D: n_spans; [r, i, j]
        """
        n_words = len(marks[0])
        self.span_list = list(comb(range(n_words), 2))
        return [self._greedy_search(score, mark)
                for score, mark in zip(scores, marks)]

    def _greedy_search(self, scores, mark):
        """
        :param scores: 1D: n_labels, 2D: n_spans; score
        :param mark: 1D: n_words; elem=0/1
        :return: 1D: n_spans, 2D: [r, i, j]
        """
        triples = []
        used_words = deepcopy(mark)
        used_labels = []

        n_words = len(mark)
        prd_index = mark.nonzero()[0][0]
        prd_span_index = convert_span_to_span_index(i=prd_index,
                                                    j=prd_index,
                                                    n_words=n_words)
        spans = self._sort_spans(scores=scores,
                                 prd_index=prd_index,
                                 prd_span_index=prd_span_index)

        for (r, i, j, _) in spans:
            if r in used_labels:
                continue
            if used_words[i: j + 1].sum() > 0:
                continue

            triples.append([r, i, j])

            used_words[i: j + 1] = 1
            if r in self.core_label_ids:
                used_labels.append(r)

        return triples

    def _sort_spans(self, scores, prd_index, prd_span_index, gamma=1.0):
        """
        :param scores: 1D: n_labels, 2D: n_spans; score
        :return: 1D: n_labels, 2D: n_words * n_words; elem=(r, i, j, score)
        """
        spans = []
        for r, scores_row in enumerate(scores):
            score_prd = gamma * scores_row[prd_span_index]
            for index, score in enumerate(scores_row):
                (i, j) = self.span_list[index]
                if i <= prd_index <= j:
                    continue
                if score_prd < score:
                    spans.append((r, i, j, score))
        spans.sort(key=lambda span: span[-1], reverse=True)
        return spans
