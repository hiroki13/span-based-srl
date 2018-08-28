import theano.tensor as T


class Regularizer(object):
    def __call__(self, **kwargs):
        raise NotImplementedError


class L2Regularizer(Regularizer):
    def __call__(self, alpha, params):
        return alpha * l2_sqr(params) / 2.


def l2_sqr(params):
    sqr = 0.0
    for p in params:
        sqr += T.sum((p ** 2))
    return sqr
