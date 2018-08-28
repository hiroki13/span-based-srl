import numpy as np
import theano
import theano.tensor as T

from utils.savers import save_pickle
from utils.loaders import load_pickle


def get_optimizer(argv):
    if argv.opt_type == 'adam':
        return Adam(argv=argv, lr=argv.lr, grad_clip=argv.grad_clip)
    return SGD(argv=argv, lr=argv.lr, grad_clip=argv.grad_clip)


class Optimizer(object):
    def __init__(self, **kwargs):
        self.argv = kwargs['argv']
        self.grad_clip = kwargs['grad_clip']
        self.params = []

    def __call__(self, grads, params):
        raise NotImplementedError

    def set_params(self, **kwargs):
        raise NotImplementedError

    def init_params(self):
        for p in self.params:
            p.set_value(p.get_value(borrow=True) * 0)

    @staticmethod
    def _grad_clipping(gradients, max_norm=5.0):
        global_grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gradients)))
        multiplier = T.switch(global_grad_norm < max_norm, 1.0, max_norm / global_grad_norm)
        return [g * multiplier for g in gradients]

    def save_params(self, epoch=0):
        argv = self.argv
        if argv.output_dir:
            dir_name = argv.output_dir
        else:
            dir_name = 'output'
        if argv.output_fn:
            file_name = '/opt.param.%s.epoch-%d' % (argv.output_fn, epoch)
        else:
            file_name = '/opt.param.%s.epoch-%d' % (argv.method, epoch)

        fn = dir_name + file_name
        params = [p.get_value(borrow=True) for p in self.params]
        save_pickle(fn=fn, data=params)

    def load_params(self, path):
        params = load_pickle(path)
        assert len(self.params) == len(params)
        for p1, p2 in zip(self.params, params):
            p1.set_value(p2)


class SGD(Optimizer):
    def __init__(self, lr=0.001, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX), borrow=True)

    def __call__(self, params, grads):
        updates = []
        if self.grad_clip:
            grads = self._grad_clipping(grads, max_norm=1.0)
        for p, g in zip(params, grads):
            updates.append((p, p - self.lr * g))
        return updates

    def set_params(self):
        pass


class Adam(Optimizer):
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.lr = theano.shared(np.asarray(lr, dtype=theano.config.floatX), borrow=True)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def __call__(self, params, grads):
        updates = []

        i = self.params[0]
        i_t = i + 1.
        a_t = self.lr * T.sqrt(1 - self.b2 ** i_t) / (1 - self.b1 ** i_t)

        if self.grad_clip:
            grads = self._grad_clipping(grads, max_norm=1.0)

        for index, (p, g) in enumerate(zip(params, grads)):
            v = self.params[2 * index + 1]
            r = self.params[2 * index + 2]
            index += 2

            v_t = self.b1 * v + (1. - self.b1) * g
            r_t = self.b2 * r + (1. - self.b2) * g ** 2

            step = a_t * v_t / (T.sqrt(r_t) + self.eps)

            updates.append((v, v_t))
            updates.append((r, r_t))
            updates.append((p, p - step))

        updates.append((i, i_t))
        return updates

    def set_params(self, params):
        i = theano.shared(np.asarray(.0, dtype=theano.config.floatX))
        self.params.append(i)
        for p in params:
            p_tm = p.get_value(borrow=True)
            v = theano.shared(np.zeros(p_tm.shape, dtype=p_tm.dtype))
            r = theano.shared(np.zeros(p_tm.shape, dtype=p_tm.dtype))
            self.params += [v, r]
