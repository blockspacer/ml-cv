# coding: utf-8

import sys
sys.path.append("pykitti_master")  # fixme: baaaad...
import pykitti

if __name__ == '__main__':
    basedir = '/mnt/d1/datasets'
    date = '2011_09_26'
    drive = '0056'
    raw = False  # True

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive, range(81, 82, 1))
    dataset.load_calib()
    c = dataset.calib
    dataset.load_gray(format='cv2')  # Loads images as uint8 grayscale

    # select
    idx = 0
    l = dataset.gray[idx].left
    r = dataset.gray[idx].right
