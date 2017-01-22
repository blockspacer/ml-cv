# encoding: utf-8

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from collections import namedtuple

# NetworkX
# https://networkx.github.io/documentation/networkx-1.9.1/reference/algorithms.component.html
# https://media.readthedocs.org/pdf/networkx/stable/networkx.pdf
# "Connected component (graph theory)"
import networkx as nx

import itertools


def flat(ll):
    return list(itertools.chain.from_iterable(ll))


Cube = namedtuple('Cube', ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'])
Vec3 = namedtuple('Vec3', ['x', 'y', 'z'])


class Cell(object):
    def __init__(self, _id, vec3):
        self.id = _id
        self.xyz = vec3
        self.hits = 0
        self.label = None


def L2_in_cells(cell0, cell1):
    l = np.array(cell0.xyz)
    r = np.array(cell1.xyz)
    return np.linalg.norm(l - r)


class Catcher(object):
    def __init__(self, cube, step_xyz=0.25):
        self.step_xyz = step_xyz
        cs = self.step_xyz
        x_size = int((cube.x_max - cube.x_min) / cs)
        y_size = int((cube.y_max - cube.y_min) / cs)
        z_size = int((cube.z_max - cube.z_min) / cs)
        self.shape = (x_size, y_size, z_size)
        self.cube = cube
        self.total_size = x_size * y_size * z_size

        self.cells = [list([]) for _ in xrange(x_size)]
        id = 0
        cells = self.cells
        for x in range(x_size):
            cells[x] = [list([]) for _ in xrange(y_size)]
            for y in range(y_size):
                cells[x][y] = [None for _ in xrange(z_size)]
                for z in range(z_size):
                    vec3 = Vec3(x, y, z)
                    cell = Cell(id, vec3)
                    cells[x][y][z] = cell
                    id += 1

    def get_yz_layer_size(self):
        return self.shape[0] * self.shape[1]

    def insert(self, vec3):
        s = self
        x = int(min(max((vec3.x - s.cube.x_min) / s.step_xyz, 0), s.shape[0] - 1))
        y = int(min(max((vec3.y - s.cube.y_min) / s.step_xyz, 0), s.shape[1] - 1))
        z = int(min(max((vec3.z - s.cube.z_min) / s.step_xyz, 0), s.shape[2] - 1))
        self.cells[x][y][z].hits += 1

    def batch_insert(self, batch):
        h, w = batch.shape
        for x in range(h):
            v3 = Vec3(*batch[x, :])
            self.insert(v3)

    def dump_to_3d(self, thr=0):
        # да, пусть 3d, нужно будет для поиска кластеров
        res = []
        s = self
        for x in range(0, self.shape[0]):
            for y in range(0, self.shape[1]):
                for z in range(0, self.shape[2]):
                    if self.cells[x][y][z].hits > thr:
                        v0 = x * s.step_xyz + s.cube.x_min + s.step_xyz / 2
                        v1 = y * s.step_xyz + s.cube.y_min + s.step_xyz / 2
                        v2 = z * s.step_xyz + s.cube.z_min + s.step_xyz / 2
                        v3 = [v0, v1, v2]
                        res.append(v3)
        return np.array(res)

    def for_each(self, fn):
        for x in range(0, self.shape[0]):
            for y in range(0, self.shape[1]):
                for z in range(0, self.shape[2]):
                    fn(self.cells[x][y][z])


def split(mat):
    return mat[:, 0], mat[:, 1], mat[:, 2]


if __name__ == '__main__':
    xlim = [-2, 15]
    ylim = [-1, 8]
    zlim = [-1, 4.5]
    lim = flat([xlim, ylim, zlim])

    cube = Cube(*lim)
    catcher = Catcher(cube)

    # Data
    # numpy.random.standard_t(df, size=None)
    mat = np.random.randn(400, 3)

    mat[:, 0] = mat[:, 0] * 2 + 1.5
    mat[:, 1] = mat[:, 1] * 0.1 + 1.5
    mat[:, 2] = mat[:, 2] * 0.1 + 1.5
    mat[:200, 2] += 1.5

    x = mat[:, 0]
    y = mat[:, 1]
    z = mat[:, 2]

    # Fill it
    # catcher.insert(Vec3(2.1, 0, 0))
    catcher.batch_insert(mat)

    # Graph
    G_index = {}


    def fn(n):
        if n.hits > 0:
            G_index[n.id] = n


    catcher.for_each(fn)

    print 'Make graph...'
    # fixme: точно можно ускорить, т.к. не нужно ставнивать все со всеми
    G = nx.Graph()
    keys = sorted(G_index.keys())
    G.add_nodes_from(keys)
    # fixme: slooow - O(n^2) if find in hash_table
    thr = 3
    yz_layer_size = catcher.get_yz_layer_size()
    for i, k1 in enumerate(keys):
        did = 0
        # fixme: ключ не должен превысить
        for k2 in keys:
            d = L2_in_cells(G_index[k1], G_index[k2])
            if d < thr:
                G.add_edge(k1, k2)

    print 'Analyse...'
    print nx.is_connected(G)
    print nx.number_connected_components(G)

    # exit(1)

    # Dump
    print 'Dump...'
    res_3d = catcher.dump_to_3d()
    x, y, z = split(res_3d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    plt.show()
