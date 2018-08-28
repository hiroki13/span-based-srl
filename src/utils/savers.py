import gzip
import pickle
import json


class Saver(object):

    def __init__(self, argv):
        self.argv = argv

    def save_props(self, **kwargs):
        raise NotImplementedError

    def save_json_format(self, **kwargs):
        raise NotImplementedError


class BaseSaver(Saver):

    def save_props(self, corpus, labels, vocab_label):
        """
        :param corpus: 1D: n_sents, 2D: n_words; elem=line
        :param labels: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label id
        :param vocab_label: Vocab()
        """
        assert len(corpus) == len(labels), '%d %d' % (len(corpus), len(labels))

        fn = self.argv.output_dir
        if self.argv.output_fn:
            fn += '/results.%s.prop' % self.argv.output_fn
        else:
            fn += '/results.prop'
        f = open(fn, 'w')

        for sent, labels_sent in zip(corpus, labels):
            columns = [[mark] for mark in sent.marks]
            for labels_prd in labels_sent:
                assert len(columns) == len(labels_prd)
                spans = self._get_spans(labels_prd, vocab_label)
                labels = self._convert_span_to_prop(len(labels_prd), spans)
                for i, label in enumerate(labels):
                    columns[i].append(label)
            for c in columns:
                f.write("%s\n" % "\t".join(c))
            f.write("\n")
        f.close()

    def save_json_format(self, corpus, labels, vocab_label):
        """
        :param corpus: 1D: n_sents, 2D: n_words; elem=line
        :param labels: 1D: n_sents, 2D: n_prds, 3D: n_words; elem=label id
        :param vocab_label: Vocab()
        """
        assert len(corpus) == len(labels), '%d %d' % (len(corpus), len(labels))

        fn = self.argv.output_dir
        if self.argv.output_fn:
            fn += '/results.%s.json' % self.argv.output_fn
        else:
            fn += '/results.json'
        f = open(fn, 'w')

        corpus_dic = {}
        for sent_index, (sent, labels_sent) in enumerate(zip(corpus, labels)):
            assert len(sent.prd_indices) == len(labels_sent)

            prop_dic = {}
            for prd_index, labels_prd in zip(sent.prd_indices, labels_sent):
                arg_dic = {}
                spans = self._get_spans(labels_prd, vocab_label)
                for (label, i, j) in spans:
                    if label == 'V':
                        continue
                    key = '(%s,%d,%d)' % (label, i, j)
                    value = " ".join(sent.strings[i: j + 1])
                    arg_dic[key] = value

                prd_dic = {'prd': sent.forms[prd_index],
                           'arg': arg_dic}
                prop_dic['prd-%d' % prd_index] = prd_dic

            sent_dic = {'text': " ".join(sent.strings),
                        'mark': " ".join(sent.marks),
                        'prop': prop_dic}
            corpus_dic['sent-%d' % sent_index] = sent_dic

        json.dump(corpus_dic, f, indent=4)
        f.close()

    @staticmethod
    def _get_spans(labels, vocab_label):
        """
        :param labels: 1D: n_words; elem=label id
        :param vocab_label: label id dict
        :return: 1D: n_spans; elem=[label, i, j]
        """
        spans = []
        span = []
        for w_i, label_id in enumerate(labels):
            label = vocab_label.get_word(label_id)
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

    @staticmethod
    def _convert_span_to_prop(n_words, spans):
        """
        :param n_words: int
        :param spans: 1D: n_spans; elem=[label, i, j]
        :return: 1D: n_words; elem=label
        """
        k = 0
        args = []
        for w_i in range(n_words):
            if k >= len(spans):
                args.append('*')
                continue
            span = spans[k]
            if span[1] < w_i < span[2]:  # within span
                args.append('*')
            elif w_i == span[1] and w_i == span[2]:  # begin and end of span
                args.append('(' + span[0] + '*)')
                k += 1
            elif w_i == span[1]:  # begin of span
                args.append('(' + span[0] + '*')
            elif w_i == span[2]:  # end of span
                args.append('*)')
                k += 1
            else:
                args.append('*')  # without span
        return args


class SpanSaver(Saver):

    def save_props(self, corpus, labels, vocab_label):
        """
        :param corpus: 1D: n_sents, 2D: n_words; elem=line
        :param labels: 1D: n_sents, 2D: n_prds, 3D: n_spans, 4D: [r, i, j]
        """
        assert len(corpus) == len(labels), '%d %d' % (len(corpus), len(labels))

        fn = self.argv.output_dir
        if self.argv.output_fn:
            fn += '/results.%s.prop' % self.argv.output_fn
        else:
            fn += '/results.prop'
        f = open(fn, 'w')

        for sent, spans_sent in zip(corpus, labels):
            columns = [[mark] for mark in sent.marks]
            n_words = sent.n_words
            assert len(sent.prd_indices) == len(spans_sent)
            for prd_index, spans in zip(sent.prd_indices, spans_sent):
                prop = self._convert_span_to_prop(spans=spans,
                                                  prd_index=prd_index,
                                                  n_words=n_words,
                                                  vocab_label=vocab_label)
                for i, p in enumerate(prop):
                    columns[i].append(p)
            for c in columns:
                f.write("%s\n" % "\t".join(c))
            f.write("\n")
        f.close()

    def save_json_format(self, corpus, labels, vocab_label):
        """
        :param corpus: 1D: n_sents, 2D: n_words; elem=line
        :param labels: 1D: n_sents, 2D: n_prds, 3D: n_spans, 4D: [r, i, j]
        :param vocab_label: Vocab()
        """
        assert len(corpus) == len(labels), '%d %d' % (len(corpus), len(labels))

        fn = self.argv.output_dir
        if self.argv.output_fn:
            fn += '/results.%s.json' % self.argv.output_fn
        else:
            fn += '/results.json'
        f = open(fn, 'w')

        corpus_dic = {}
        for sent_index, (sent, spans_sent) in enumerate(zip(corpus, labels)):
            assert len(sent.prd_indices) == len(spans_sent)

            prop_dic = {}
            for prd_index, spans in zip(sent.prd_indices, spans_sent):
                arg_dic = {}
                for (r, i, j) in spans:
                    key = '(%s,%d,%d)' % (vocab_label.get_word(r), i, j)
                    value = " ".join(sent.strings[i: j + 1])
                    arg_dic[key] = value

                prd_dic = {'prd': sent.forms[prd_index],
                           'arg': arg_dic}
                prop_dic['prd-%d' % prd_index] = prd_dic

            sent_dic = {'text': " ".join(sent.strings),
                        'mark': " ".join(sent.marks),
                        'prop': prop_dic}
            corpus_dic['sent-%d' % sent_index] = sent_dic

        json.dump(corpus_dic, f, indent=4)
        f.close()

    @staticmethod
    def _convert_span_to_prop(spans, prd_index, n_words, vocab_label):
        """
        :param spans: 1D: n_spans, 2D: [r, i, j]
        :return: 1D: n_words; elem=str; e.g. (A0* & *)
        """
        prop = ['*' for _ in range(n_words)]
        prop[prd_index] = '(V*)'
        for (label_id, pre_index, post_index) in spans:
            label = vocab_label.get_word(label_id)
            if pre_index == post_index:
                prop[pre_index] = '(%s*)' % label
            else:
                prop[pre_index] = '(%s*' % label
                prop[post_index] = '*)'
        return prop


def save_pickle(fn, data):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        pickle.dump(data, gf, pickle.HIGHEST_PROTOCOL)


def save_key_value_format(fn, keys, values):
    assert len(keys) == len(values)
    if type(values[0]) is not str:
        values = map(lambda v: str(v), values)
    with open(fn + '.txt', 'w') as f:
        for key, value in zip(keys, values):
            f.write("%s\t%s\n" % (key, value))
