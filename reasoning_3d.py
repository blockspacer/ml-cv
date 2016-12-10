# encoding: utf-8

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 20
c, m, zl, zh = ('r', 'o', -5, 25)
xs = randrange(n, 23, 32)
ys = randrange(n, 0, 100)
zs = randrange(n, zl, zh)
ax.scatter(xs, ys, zs, c=c, marker=m)

c, m, zl, zh = ('b', '^', -7+70, -5+70)
xs = randrange(n, 23, 32)
ys = randrange(n, 0, 200)
zs = randrange(n, zl, zh)
ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# fixme: а можно распознавать по карте диспарити

# Trouble:
# - Виден не весь объект, CAD модели как синтетические
# - не знаем сколько объектов

# Solutions:
# fuzzing....
