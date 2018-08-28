import gzip
import pickle
import json


class Saver(object):
    def __init__(self, argv):
        self.argv = argv

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
