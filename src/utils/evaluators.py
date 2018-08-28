import numpy as np

from utils.misc import write, convert_span_to_span_index


class Evaluator(object):
    def __init__(self, argv):
        self.argv = argv

    def f_score(self, y_true, y_pred, vocab_label):
        """
        :param y_true: 1D: n_batches, 2D: batch_size, 3D: n_words; elem=label id
        :param y_pred: 1D: n_batches, 2D: batch_size, 3D: n_words; elem=label id
        :param vocab_label: Vocab()
        """
        correct, p_total, r_total = self.calc_metrics(y_true=y_true,
                                                      y_pred=y_pred,
                                                      vocab_label=vocab_label)
        p, r, f = calc_f_score(correct, p_total, r_total)
        write('\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
            f, p, int(correct), int(p_total), r, int(correct), int(r_total)))
        return f

    def calc_metrics(self, y_true, y_pred, vocab_label):
        p_total = 0.
        r_total = 0.
        correct = 0.
        for y_true_batch, y_pred_batch in zip(y_true, y_pred):
            for y_true_i, y_pred_i in zip(y_true_batch, y_pred_batch):
                y_true_spans = get_spans_from_bio_labels(y_true_i, vocab_label)
                y_pred_spans = get_spans_from_bio_labels(y_pred_i, vocab_label)
                p_total += len(y_pred_spans)
                r_total += len(y_true_spans)
                for y_pred_span in y_pred_spans:
                    if y_pred_span in y_true_spans:
                        correct += 1.
        return correct, p_total, r_total


class SpanEvaluator(Evaluator):
    def f_score(self, y_true, y_pred, vocab_label):
        """
        :param y_true: 1D: n_batches, 2D: batch_size, 3D: n_spans, 4D: [label_id, pre_index, post_index]
        :param y_pred: 1D: n_batches, 2D: batch_size, 3D: n_spans, 4D: [label_id, pre_index, post_index]
        """
        correct, p_total, r_total = self.calc_metrics(y_true=y_true,
                                                      y_pred=y_pred,
                                                      vocab_label=vocab_label)
        p, r, f = calc_f_score(correct, p_total, r_total)
        write('\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
            f, p, int(correct), int(p_total), r, int(correct), int(r_total)))
        return f

    def calc_metrics(self, y_true, y_pred, vocab_label):
        """
        :param y_true: 1D: n_batches, 2D: batch_size, 3D: n_spans, 4D: [label_id, pre_index, post_index]
        :param y_pred: 1D: n_batches, 2D: batch_size, 3D: n_spans, 4D: [label_id, pre_index, post_index]
        """
        p_total = 0.
        r_total = 0.
        correct = 0.
        for span_true_batch, span_pred_batch in zip(y_true, y_pred):
            for spans_true, spans_pred in zip(span_true_batch, span_pred_batch):
                spans_true = self._concat_c_spans_from_spans(spans_true, vocab_label)
                spans_pred = self._concat_c_spans_from_spans(spans_pred, vocab_label)
                p_total += len(spans_pred)
                r_total += len(spans_true)
                for span in spans_pred:
                    if span in spans_true:
                        correct += 1
        return correct, p_total, r_total

    @staticmethod
    def _concat_c_spans_from_spans(spans, vocab_label):
        spans = [[vocab_label.get_word(l), i, j] for (l, i, j) in spans]
        labels = [l for (l, i, j) in spans]
        c_indices = [index for index, (l, i, j) in enumerate(spans) if l.startswith('C')]
        non_ant_c_spans = []

        for c_index in c_indices:
            c_span = spans[c_index]
            label = c_span[0][2:]
            if label in labels:
                spans[labels.index(label)].extend(c_span[1:])

        concated_spans = [span for i, span in enumerate(spans) if i not in c_indices]
        spans = concated_spans + non_ant_c_spans
        return spans


def calc_f_score(correct, p_total, r_total):
    precision = correct / p_total if p_total > 0 else 0.
    recall = correct / r_total if r_total > 0 else 0.
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.
    return precision, recall, f1


def calc_metrics_for_bio(y_true, y_pred, vocab_label):
    p_total = 0.
    r_total = 0.
    correct = 0.
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        y_true_spans = get_spans_from_bio_labels(y_true_i, vocab_label)
        y_pred_spans = get_spans_from_bio_labels(y_pred_i, vocab_label)
        p_total += len(y_pred_spans)
        r_total += len(y_true_spans)
        for y_pred_span in y_pred_spans:
            if y_pred_span in y_true_spans:
                correct += 1.
    return correct, p_total, r_total


def get_spans_from_bio_labels(sent, vocab_label):
    spans = []
    span = []
    for w_i, label_id in enumerate(sent):
        label = vocab_label.get_word(label_id)
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

    return concat_c_spans_from_bio_labels(spans)


def concat_c_spans_from_bio_labels(spans):
    labels = [span[0] for span in spans]
    c_indices = [i for i, span in enumerate(spans) if span[0].startswith('C')]
    non_ant_c_spans = []

    for c_index in c_indices:
        c_span = spans[c_index]
        c_label = c_span[0][2:]
        if c_label in labels:
            spans[labels.index(c_label)].extend(c_span[1:])
        else:
            non_ant_c_spans.append([c_label] + c_span[1:])
    concated_spans = [span for i, span in enumerate(spans) if i not in c_indices]
    spans = concated_spans + non_ant_c_spans
    return spans


def calc_metrics_for_spans(span_true, span_pred, marks):
    """
    :param span_true: 1D: batch_size * n_spans, 2D: [batch_index, label_id, pre_index, post_index]
    :param span_pred: 1D: batch_size, 2D: n_labels, 3D: [pre_index, post_index]
    :param marks: 1D: batch_size, 2D: n_words; elem=0/1
    :return:
    """
    correct = 0.
    p_total = 0.
    r_total = 0.
    for b_index, (span_pred_i, marks_i) in enumerate(zip(span_pred, marks)):
        prd_index = list(marks_i).index(1)
        for label_id, span_p in enumerate(span_pred_i):
            pre_index, post_index = span_p
            if not (pre_index == post_index == prd_index):
                p_total += 1
                if [b_index, label_id, pre_index, post_index] in span_true:
                    correct += 1

    for tuples in span_true:
        b_index, label_id, pre_index, post_index = tuples
        prd_index = list(marks[b_index]).index(1)
        if not (pre_index == post_index == prd_index):
            r_total += 1

    return correct, p_total, r_total


def calc_correct_and_pred_spans(span_true, span_pred, marks):
    """
    :param span_true: 1D: batch_size * n_spans; span index
    :param span_pred: 1D: batch_size, 2D: n_labels; span index
    :param marks: 1D: batch_size, 2D: n_words; elem=0/1
    """
    correct = 0.
    n_pred_spans = 0.
    n_words = len(marks[0])
    _, prd_indices = np.array(marks).nonzero()
    prd_indices = [convert_span_to_span_index(p, p, n_words)
                   for p in prd_indices]

    for b_index, span_pred_tmp in enumerate(span_pred):
        prd_index = prd_indices[b_index]
        for label_id, span_index in enumerate(span_pred_tmp):
            if span_index == prd_index:
                continue
            if [b_index, label_id, span_index] in span_true:
                correct += 1
            n_pred_spans += 1

    return correct, n_pred_spans


def calc_true_spans(sents):
    """
    :param sents: 1D: n_sents
    :return: total number of spans
    """
    return sum([len(triple) for sent in sents for triple in sent.span_triples])
