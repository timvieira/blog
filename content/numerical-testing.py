import numpy as np
import pylab as pl
from collections import defaultdict

from numpy import log, exp, expm1, log1p
import mpmath


def exp_normalize1(a):
    y = np.exp(a)
    y /= y.sum()
    return y


def exp_normalize2(a):
    y = np.exp(a - a.max())
    y /= y.sum()
    return y


def test():

    method = defaultdict(list)

    for ww in np.linspace(0, 1000, 100):
        w = [0, ww]

        x = np.array(w, dtype=np.float128)
        a = exp_normalize1(x)
        b = exp_normalize2(x)
        reference = b

        assert (np.abs(a - b) <= 1e-15).all()

        x = np.array(w, dtype=np.float64)
        #x = np.array(w, dtype=np.float32)
        #x = np.array(w, dtype=np.float16)

        a = exp_normalize1(x)
        b = exp_normalize2(x)

        print 'agree: %g a: %g b: %g' \
            % (np.abs(a-b).sum(),
               np.abs(a-reference).sum(),
               np.abs(b-reference).sum())

        err = np.abs(a-reference)
        err[np.isnan(err)] = 1.0
        method['a'].append((ww, err.mean()))

        err = np.abs(b-reference)
        err[np.isnan(err)] = 1.0
        method['b'].append((ww, err.mean()))

    for name, data in method.items():
        x, y = zip(*data)
        pl.plot(x, y, lw=1, alpha=0.5, label=name)

    pl.legend()
    pl.show()

#from numpy import log, exp, log1p
#def f1(x):
#    return log(1 + exp(-x))
#def f2(x):
#    return log1p(exp(x))
#def f3(x):
#    return x + exp(-x)
#def f4(x):
#    return x


def test2():

    method = defaultdict(list)

#    for w in np.logspace(1e-18, 1e+2, 1000):

    for w in [10**i for i in np.arange(-17, 3, .01)]:
        #x = np.array(w, dtype=np.float128)

        fs = [
#            'log(1 + exp(-x))',
#            'log1p(exp(x))',
#            'x + exp(-x)',
#            'x'
            'log(1-exp(-x))',
            'log(-expm1(-x))',
            'log1p(-exp(-x))',
        ]

        # log1p(x) = log(1+x)
        # expm1(x) = exp(x) - 1

        # None of the following three expressions uniformly sufficent:
        #   log(1-exp(-x))
        #     = log(-expm1(-x))
        #     = log1p(-exp(-x))

        #mpmath.x =
        #mpmath.expm1 = lambda x: mpmath.exp(x)-1

        #reference = float(eval(fs[0], mpmath.__dict__))

        x = w
        reference = mpmath.log(mpmath.mpf(1) - mpmath.exp(-mpmath.mpf(x)))

        print
        print '%g %r' % (w, reference)

        sty = dict(zip(fs,
                       [{'c': 'b'},
                        {'c': 'k'},
                        {'c': 'r'}]))

        for f in fs:

            x = np.array(w, dtype=np.float64)
            #x = np.array(w, dtype=np.float32)
            #x = np.array(w, dtype=np.float16)

            y = eval(f)
            err = np.abs(y-reference) #/ abs(reference)

            print f, repr(y), err
            method[f].append((w, err))

        print

    for name, data in method.items():
        x, y = zip(*data)
        pl.plot(x, y, lw=1, alpha=0.5, label=name, **sty[name])
    pl.xscale('log')
    pl.yscale('log')
    pl.ylim(1e-20, 1e-1)
    pl.legend()
    pl.show()


if __name__ == '__main__':
    #test()
    test2()
