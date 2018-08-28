import numpy as np
import theano


class Initializer(object):
    def __call__(self, shape, shared=True, name=None):
        raise NotImplementedError


class Zero(Initializer):
    def __call__(self, shape, shared=True, name=None):
        param = np.zeros(shape, theano.config.floatX)
        if shared:
            return theano.shared(value=param, name=name, borrow=True)
        return param


class One(Initializer):
    def __call__(self, shape, shared=True, name=None):
        param = np.ones(shape, theano.config.floatX)
        if shared:
            return theano.shared(value=param, name=name, borrow=True)
        return param


class Identity(Initializer):
    def __call__(self, shape, shared=True, name=None):
        assert len(shape) == 2
        param = np.ones(shape[0], theano.config.floatX)
        param = np.diag(param)
        if shared:
            return theano.shared(value=param, name=name, borrow=True)
        return param


class Uniform(Initializer):
    def __call__(self, shape, shared=True, name=None):
        param = np.asarray(np.random.uniform(low=-0.01,
                                             high=0.01,
                                             size=shape),
                           dtype=theano.config.floatX)
        if shared:
            return theano.shared(value=param, name=name, borrow=True)
        return param


class Normal(Initializer):
    def __call__(self, shape, shared=True, name=None):
        param = np.asarray(np.random.normal(0.0, 0.01, shape),
                           dtype=theano.config.floatX)
        if shared:
            return theano.shared(value=param, name=name, borrow=True)
        return param


class Xavier(Initializer):
    def __call__(self, shape, shared=True, name=None):
        param = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / np.sum(shape)),
                                             high=np.sqrt(6.0 / np.sum(shape)),
                                             size=shape),
                           dtype=theano.config.floatX)
        if shared:
            return theano.shared(value=param, name=name, borrow=True)
        return param


class Orthonormal(Initializer):
    """
    This is based on the implementation of Luheng He;
    https://github.com/luheng/deep_srl
    """
    def __call__(self, shape, shared=True, name=None):
        assert len(shape) == 2
        if shape[0] == shape[1]:
            M = np.random.randn(*shape).astype(theano.config.floatX)
            Q, R = np.linalg.qr(M)
            Q = Q * np.sign(np.diag(R))
            param = Q * 1.0
        else:
            M1 = np.random.randn(shape[0], shape[0]).astype(theano.config.floatX)
            M2 = np.random.randn(shape[1], shape[1]).astype(theano.config.floatX)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            param = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * 1.0
        if shared:
            return theano.shared(value=param, name=name, borrow=True)
        return param
