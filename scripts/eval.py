import sys

import numpy as np


def load(path, data_size=100000000):
    corpus = []
    sent = []
    with open(path) as f:
        for line in f:
            elem = [l for l in line.rstrip().split()]
            if len(elem) > 0:
                sent.append(elem)
            else:
                corpus.append(sent)
                sent = []
            if len(corpus) >= data_size:
                break
    return corpus


def f_score(crr_total, p_total, r_total):
    precision = crr_total / p_total if p_total > 0 else 0.
    recall = crr_total / r_total if r_total > 0 else 0.
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.
    return precision, recall, f1


def accuracy(crr_total, total):
    return crr_total / total if total > 0 else 0.


def srl_metrics(y_true, y_pred):
    """
    :param y_true: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label
    :param y_pred: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label
    """
    p_total = 0.
    r_total = 0.
    crr_total = 0.

    assert len(y_true) == len(y_pred)
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        assert len(y_true_i) == len(y_pred_i)
        for y_true_j, y_pred_j in zip(y_true_i[1:], y_pred_i[1:]):
            assert len(y_true_j) == len(y_pred_j)
            y_true_spans = get_labeled_spans(y_true_j)
            y_pred_spans = get_labeled_spans(y_pred_j)
            p_total += len(y_pred_spans)
            r_total += len(y_true_spans)
            for y_pred_span in y_pred_spans:
                if y_pred_span in y_true_spans:
                    crr_total += 1.
    return crr_total, p_total, r_total


def span_metrics(y_true, y_pred):
    """
    :param y_true: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label
    :param y_pred: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label
    """
    p_total = 0.
    r_total = 0.
    crr_total = 0.

    assert len(y_true) == len(y_pred)
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        assert len(y_true_i) == len(y_pred_i)
        for y_true_j, y_pred_j in zip(y_true_i[1:], y_pred_i[1:]):
            assert len(y_true_j) == len(y_pred_j)
            y_true_spans = get_labeled_spans(y_true_j)
            y_pred_spans = get_labeled_spans(y_pred_j)
            p_total += len(y_pred_spans)
            r_total += len(y_true_spans)

            y_true_boundary = [span[1:] for span in y_true_spans]
            for y_pred_span in y_pred_spans:
                if y_pred_span[1:] in y_true_boundary:
                    crr_total += 1.
    return crr_total, p_total, r_total


def label_metrics(y_true, y_pred):
    """
    :param y_true: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label
    :param y_pred: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label
    """
    total = 0.
    crr_total = 0.

    assert len(y_true) == len(y_pred)
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        assert len(y_true_i) == len(y_pred_i)
        for y_true_j, y_pred_j in zip(y_true_i[1:], y_pred_i[1:]):
            assert len(y_true_j) == len(y_pred_j)
            y_true_spans = get_labeled_spans(y_true_j)
            y_pred_spans = get_labeled_spans(y_pred_j)

            y_true_boundary = [span[1:] for span in y_true_spans]
            for y_pred_span in y_pred_spans:
                if y_pred_span[1:] in y_true_boundary:
                    total += 1.
                    index = y_true_boundary.index(y_pred_span[1:])
                    y_true_span = y_true_spans[index]
                    if y_pred_span[0] == y_true_span[0]:
                        crr_total += 1.
    return crr_total, total


def srl_metrics_per_distance(y_true, y_pred):
    """
    :param y_true: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label
    :param y_pred: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label
    """
    def _dist(i_, j_, prd_index_):
        if j_ < prd_index_:
            return prd_index_ - j_ - 1
        return i_ - prd_index_ - 1

    def _dist_bin(dist):
        if dist == 0:
            return 0
        elif 0 < dist < 3:
            return 1
        elif 3 <= dist < 7:
            return 2
        return 3

    dist_dict = np.zeros(shape=(4, 3), dtype="float32")

    assert len(y_true) == len(y_pred)
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        assert len(y_true_i) == len(y_pred_i)

        prds = y_true_i[0]
        prd_indices = [i for i, y in enumerate(prds) if y != "-"]

        for y_true_j, y_pred_j, prd_index in zip(y_true_i[1:], y_pred_i[1:], prd_indices):
            assert len(y_true_j) == len(y_pred_j)
            y_true_spans = get_labeled_spans(y_true_j)
            y_pred_spans = get_labeled_spans(y_pred_j)

            for span in y_true_spans:
                # Remove continuous spans
                if len(span) > 3:
                    continue
                (label, i, j) = span
                dist = _dist(i, j, prd_index)
                binned_dist = _dist_bin(dist)
                dist_dict[binned_dist][2] += 1

            for span in y_pred_spans:
                if len(span) > 3:
                    continue
                (label, i, j) = span
                dist = _dist(i, j, prd_index)
                binned_dist = _dist_bin(dist)
                dist_dict[binned_dist][1] += 1

            for y_pred_span in y_pred_spans:
                if y_pred_span in y_true_spans:
                    if len(y_pred_span) > 3:
                        continue
                    label, i, j = y_pred_span
                    dist = _dist(i, j, prd_index)
                    binned_dist = _dist_bin(dist)
                    dist_dict[binned_dist][0] += 1

    return dist_dict


def get_labeled_spans(prop):
    """
    :param prop: 1D: n_words; elem=bracket label
    :return: 1D: n_words; elem=BIO label
    """
    def _concat_c_spans(_spans):
        labels = [_span[0] for _span in _spans]
        c_indices = [i for i, _span in enumerate(_spans) if _span[0].startswith('C')]
        non_ant_c_spans = []

        for c_index in c_indices:
            c_span = _spans[c_index]
            _label = c_span[0][2:]
            if _label in labels:
                _spans[labels.index(_label)].extend(c_span[1:])
            else:
                non_ant_c_spans.append([_label] + c_span[1:])
        concated_spans = [span for i, span in enumerate(_spans) if i not in c_indices]
        _spans = concated_spans + non_ant_c_spans
        return _spans

    labeled_spans = []
    labeled_span = []
    for i, arg in enumerate(prop):
        if arg.startswith('('):
            if arg.endswith(')'):
                label = arg.split("*")[0][1:]
                labeled_span = [label, i, i]
            else:
                label = arg[1:-1]
                labeled_span = [label, i]
        elif arg.endswith(')'):
            labeled_span.append(i)

        if len(labeled_span) == 3 and labeled_span[0] != "V" and labeled_span[0] != "C-V":
            labeled_spans.append(labeled_span)
            labeled_span = []

    labeled_spans = _concat_c_spans(labeled_spans)
    return labeled_spans


def print_metrics(y_true, y_pred):
    """
    :param y_true: 1D: n_sents, 2D: n_words, 3D: n_prds; elem=label
    :param y_pred: 1D: n_sents, 2D: n_words, 3D: n_prds; elem=label
    """
    crr_total, p_total, r_total = srl_metrics(y_true, y_pred)
    p, r, f = f_score(crr_total, p_total, r_total)
    sys.stdout.write('SRL RESULTS\n\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
        f, p, int(crr_total), int(p_total), r, int(crr_total), int(r_total)))
    sys.stdout.flush()

    crr_total, p_total, r_total = span_metrics(y_true, y_pred)
    p, r, f = f_score(crr_total, p_total, r_total)
    sys.stdout.write('SPAN BOUNDARY MATCH\n\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
        f, p, int(crr_total), int(p_total), r, int(crr_total), int(r_total)))
    sys.stdout.flush()

    crr_total, total = label_metrics(y_true, y_pred)
    acc = accuracy(crr_total, total)

    sys.stdout.write('LABEL MATCH\n\tACCURACY:{:>7.2%} ({:>5}/{:>5})\n'.format(
        acc, int(crr_total), int(total)))
    sys.stdout.flush()


def print_metrics_per_dist(y_true, y_pred):
    metric_matrix = srl_metrics_per_distance(y_true, y_pred)
    sys.stdout.write('SRL RESULTS PER DISTANCE (C-LABEL removed)\n')
    for i, metric in enumerate(metric_matrix):
        crr_total, p_total, r_total = metric
        if i == 0:
            dist = '0'
        elif i == 1:
            dist = '1-2'
        elif i == 2:
            dist = '3-6'
        else:
            dist = '7-max'

        p, r, f = f_score(crr_total, p_total, r_total)
        sys.stdout.write('\t{}\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
            dist, f, p, int(crr_total), int(p_total), r, int(crr_total), int(r_total)))
        sys.stdout.flush()


def main(argv):
    sys.stdout.write("\nEVALUATION START\n")
    sys.stdout.flush()

    sents1 = load(argv[1])
    sents2 = load(argv[2])

    sents1 = [list(zip(*sent)) for sent in sents1]
    sents2 = [list(zip(*sent)) for sent in sents2]

    print_metrics(sents1, sents2)
    print_metrics_per_dist(sents1, sents2)


if __name__ == '__main__':
    main(sys.argv)
