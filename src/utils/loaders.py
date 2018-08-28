import os
import gzip
import pickle
import h5py

import numpy as np
import theano

from utils.misc import get_file_names_in_dir
from utils.vocab import UNK


class Loader(object):
    def __init__(self, argv):
        self.argv = argv

    def load(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_data(fn):
        with gzip.open(fn, 'rb') as gf:
            return pickle.load(gf)

    @staticmethod
    def load_key_value_format(fn):
        data = []
        with open(fn, 'r') as f:
            for line in f:
                key, value = line.rstrip().split()
                data.append((key, int(value)))
        return data

    @staticmethod
    def load_hdf5(path):
        return h5py.File(path, 'r')

    def load_txt_from_dir(self, dir_path, file_prefix):
        file_names = get_file_names_in_dir(dir_path + '/*')
        file_names = [fn for fn in file_names
                      if os.path.basename(fn).startswith(file_prefix)
                      and fn.endswith('txt')]
        return [self.load(path=fn) for fn in file_names]

    def load_hdf5_from_dir(self, dir_path, file_prefix):
        file_names = get_file_names_in_dir(dir_path + '/*')
        file_names = [fn for fn in file_names
                      if os.path.basename(fn).startswith(file_prefix)
                      and fn.endswith('hdf5')]
        return [self.load_hdf5(fn) for fn in file_names]


class Conll05Loader(Loader):
    def load(self, path, data_size=1000000, is_test=False):
        if path is None:
            return []

        corpus = []
        sent = []

        with open(path) as f:
            for line in f:
                elem = [l for l in line.rstrip().split()]
                if len(elem) > 0:
                    if is_test:
                        sent.append(elem[:6])
                    else:
                        sent.append(elem)
                else:
                    corpus.append(sent)
                    sent = []
                if len(corpus) >= data_size:
                    break
        return corpus


class CoNLL12Loader(Loader):
    def load(self, path, data_size=1000000, is_test=False):
        if path is None:
            return []

        corpus = []
        sent = []

        with open(path) as f:
            for line in f:
                elem = [l for l in line.rstrip().split()]
                if len(elem) > 10:
                    if is_test:
                        sent.append(elem[:11])
                    else:
                        sent.append(elem)
                elif len(elem) == 0:
                    corpus.append(sent)
                    sent = []
                if len(corpus) >= data_size:
                    break
        return corpus


def load_emb(path):
    word_list = []
    emb = []
    with open(path) as f:
        for line in f:
            line = line.rstrip().split()
            word_list.append(line[0])
            emb.append(line[1:])
    emb = np.asarray(emb, dtype=theano.config.floatX)

    if UNK not in word_list:
        word_list = [UNK] + word_list
        unk_vector = np.mean(emb, axis=0)
        emb = np.vstack((unk_vector, emb))

    return word_list, emb


def load_pickle(fn):
    with gzip.open(fn, 'rb') as gf:
        return pickle.load(gf)


def load_key_value_format(fn):
    data = []
    with open(fn, 'r') as f:
        for line in f:
            key, value = line.rstrip().split()
            data.append((key, int(value)))
    return data
