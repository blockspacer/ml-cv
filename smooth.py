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


def ro(x, xi):
    return abs(x - xi)


def K(x, h):
    return np.exp(-2 * x / h)


def Kq(x, xs):
    denom = (6 * np.median(xs))
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
        self.gamma = np.ones(self.xs.shape)
        self.max_gamma = max_gamma

    def estimate(self, x, excluded=-1):
        num = 0
        denum = 0
        for i, xi in enumerate(self.xs):
            if excluded != -1:
                if excluded == i:
                    continue

            yi = self.ys[i]
            wi = K(ro(x, xi), self.hs[i])
            gi = self.gamma[i]
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

        for i, _ in enumerate(self.gamma):
            g = Kq(eps[i], eps)
            self.gamma[i] = clamp_max(1 / g, self.max_gamma)


if __name__ == '__main__':
    shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
    ys = np.random.gamma(shape, scale, 40)
    xs = xrange(len(ys))
    ys[3] = 200

    nw = RobustNadarayaWatson(xs, ys, 10)

    for _ in range(10):
        nw.iterate()
        print np.array(nw.gamma * 100, dtype=np.int)

    print "estimation..."
    ys_est = []
    for x in xs:
        y_est = nw.estimate(x)
        ys_est.append(y_est)

    print "plotting..."
    plt.plot(xs, ys)
    plt.plot(xs, ys_est, "-v")
    plt.grid()
    plt.show()
