# encoding: utf-8

import matplotlib.pyplot as plt

import numpy as np


def clamp_denom(x, eps=0.0001):
    if abs(x) < eps:
        x = eps
    return x


def clamp_max(x, maxx):
    if x > maxx:
        x = maxx
    return x


def ro(x, xi, arg=0):
    # delta = 2.1
    delta = arg
    a = x - xi
    return delta ** 2 * (np.sqrt(1 + (a / delta) ** 2) - 1)
    # return abs(x - xi)


def K(x, h):
    return np.exp(-2 * x / h)


def Kq(x, xs):
    denom = (6 * np.median(xs))
    print xs
    print "denom:", denom
    return x / clamp_denom(denom)


# https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/lecture/1ckPk/mietrichieskiie-mietody-klassifikatsii-v-zadachie-vosstanovlieniia-rieghriessii
class RobustNadarayaWatson(object):
    # fixme: add variable h
    def __init__(self, xs, ys, max_gamma=10):
        self.xs = np.array(xs)

        # calc deltas
        self.hs = np.zeros(self.xs.shape)
        for i, xi in enumerate(self.xs):
            if i == 0:
                continue
            self.hs[i - 1] = xs[i] - xs[i - 1]
        self.hs[-1] = self.hs[-2]

        self.ys = np.array(ys)
        self.gammas = np.ones(self.xs.shape)
        self.max_gamma = max_gamma

    def estimate(self, x, excluded=-1):
        num = 0
        denum = 0
        for i, xi in enumerate(self.xs):
            if excluded != -1:
                if excluded == i:
                    continue

            yi = self.ys[i]
            wi = K(ro(x, xi, self.hs[i]), self.hs[i])
            gi = self.gammas[i]
            wi *= gi

            num += yi * wi
            denum += wi

        return num / clamp_denom(denum)

    def iterate(self):
        eps = np.zeros(self.xs.shape)
        for i, xi in enumerate(self.xs):
            yi = self.ys[i]
            epi = abs(self.estimate(xi, excluded=i) - yi)
            eps[i] = epi

        for i, _ in enumerate(self.gammas):
            g = Kq(eps[i], eps)
            self.gammas[i] = clamp_max(1 / g, self.max_gamma)


if __name__ == '__main__':
    # shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
    # ys = np.random.gamma(shape, scale, 40)
    # xs = xrange(len(ys))
    # ys[3] = 200
    # dys = np.array(xrange(0, len(ys)))
    # ys += dys

    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
          30, 31, 32, 33, 34, 35, 36, 37, 38, 39, ]
    ys = [6.62978140876, 4.49678706376, 4.01898549999, 203.0, 9.71220875091, 6.81199054582, 10.3687737888,
          9.34003423625, 11.0943398334, 10.0852958726, 13.595412343, 12.4766620769, 14.2148696106, 15.7052702602,
          16.7081265738, 17.2904090912, 26.0336394742, 21.5965172674, 20.6046784136, 22.8558238898, 22.7599266781,
          22.9025721765, 27.6978857255, 26.0563418143, 27.3580205328, 26.9803286751, 28.9957041814, 28.7742286869,
          30.8504571937, 33.9582208526, 32.9446982772, 32.3383795971, 36.2575900683, 33.1961117655, 38.4606023971,
          36.6657461786, 38.2616113453, 43.806062633, 42.1742207406, 46.7033592699, ]

    for i, y in enumerate(ys):
        x = xs[i]
        print x, ",",
    print
    for i, y in enumerate(ys):
        x = xs[i]
        print y, ",",
    print

    nw = RobustNadarayaWatson(xs, ys, 10)

    for _ in range(2):
        nw.iterate()
        print np.array(nw.gammas * 100, dtype=np.int)

    print "estimation..."
    ys_est = []
    for x in xs:
        y_est = nw.estimate(x)
        ys_est.append(y_est)

    print ys_est

    print "plotting..."
    plt.plot(xs, ys)
    plt.plot(xs, ys_est, "-v")
    plt.grid()
    plt.show()
