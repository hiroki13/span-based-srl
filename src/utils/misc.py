import os
import sys
import glob

import numpy as np
import theano

from utils.vocab import Vocab


def write(s, stream=sys.stdout):
    stream.write(s + '\n')
    stream.flush()


def show_score_history(history, memo=''):
    write('F1 HISTORY' + memo)
    for k, v in sorted(history.items()):
        epoch_tm = '\t- EPOCH-{:d}  '.format(k)
        if len(v) == 1:
            f1_valid = '\tBEST VALID {:>7.2%}'.format(v[0])
            write(epoch_tm + f1_valid)
        else:
            v1, v2 = v
            f1_valid = '\tBEST VALID {:>7.2%}'.format(v1)
            f1_evalu = '\tEVALU {:>7.2%}'.format(v2)
            write(epoch_tm + f1_valid + f1_evalu)


def str_to_id(sent, vocab, unk):
    """
    :param sent: 1D: n_words
    :param vocab: Vocab()
    :return: 1D: n_words; elem=id
    """
    return list(map(lambda w: vocab.get_id(w) if vocab.has_key(w) else vocab.get_id(unk), sent))


def make_vocab_from_ids(key_value_format):
    vocab = Vocab()
    for key, value in key_value_format:
        vocab.add_word(key)
    return vocab


def array(sample, is_float=False):
    if is_float:
        return np.asarray(sample, dtype=theano.config.floatX)
    return np.asarray(sample, dtype='int32')


def average_vector(emb):
    return np.mean(np.asarray(emb[2:], dtype=theano.config.floatX), axis=0)


def unit_vector(vecs, axis):
    return vecs / np.sqrt(np.sum(vecs ** 2, axis=axis, keepdims=True))


def make_output_dir(argv):
    if argv.output_dir:
        output_dir = argv.output_dir
    else:
        output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)


def join_dir_and_file_names(dir_name, file_name):
    return os.path.join(dir_name, file_name)


def get_file_names_in_dir(dir_path, prefix=None, suffix=None):
    file_names = glob.glob(dir_path + '/*')
    if prefix:
        file_names = [fn for fn in file_names
                      if os.path.basename(fn).startswith(prefix)]
    if suffix:
        file_names = [fn for fn in file_names
                      if fn.endswith(suffix)]
    return file_names


def get_latest_param_fn(file_names):
    latest_epoch = -1
    latest_fn = None
    for fn in file_names:
        for elem in fn.split('.'):
            if elem.startswith('epoch'):
                epoch = int(elem[6:])
                if latest_epoch < epoch:
                    latest_epoch = epoch
                    latest_fn = fn
                    break
    assert latest_fn is not None
    return latest_fn, latest_epoch


def span_to_span_index(i, j, n_words):
    return i * (n_words - 1) + j - np.arange(i).sum()
