# encoding: utf-8

import numpy as np
import numpy

from numpy.linalg import norm


if __name__ == '__main__':
    dims = np.array([2, 2, 4, 3])
    raw = np.array(xrange(0, dims.prod()))
    arr = np.reshape(raw, dims)

    # https://stackoverflow.com/questions/28491230/indexing-a-numpy-array-with-a-list-of-tuples
    access_tuple = (0,0,0,1) 
    #access_tuple = (0,0,0,:)  # error!
    print arr[:, 0, 0,0]
    print arr[access_tuple]