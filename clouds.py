# coding: utf-8


import numpy as np

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))


if __name__ == '__main__':
    # https://www.cse.buffalo.edu/~jing/cse601/fa12/materials/clustering_density.pdf
    # DBSCAN
    # https://blog.dominodatalab.com/topology-and-density-based-clustering/
    #
    # FIXME: search dist modes
    # Troubles: распределение схлопнутое по Z машины будет двурогим, non-gaussian
    # Распределение воль линии возможно будет неравномерным
    # Trouble: Как быть с нагоняющиими машинами
    # 1D dbscan


    pass
