#!/usr/bin/env python
# encoding: utf-8

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''

# Python 2/3 compatibility
# from __future__ import print_function

import numpy as np
import cv2

# local modules
# from common import splitfn

# built-in modules
import os

import sys
import getopt
from glob import glob

# fixme: Attention!!!
# "Я думаю что вы знаете, что исправление дисторсии — компенсирует прямые, но искажает расстояние."

def global_algorithm():
    # fixme: калбировка отдельных камер
    # K - можно откалибровать отдельно!!! это свойства камеры, если фокус фиксированный
    # Дистросию тоже отдельно
    pass

    # Все остальное уже на месте
    pass

    # fixme: stereoCalibrate() - всей пары
    pass

if __name__ == '__main__':
    np.set_printoptions(precision=4)

    p = '/home/zaqwes/datasets/2011_09_26/2011_09_26_drive_0119_extract/image_00/data'
    img_names = glob( p+"/*.png" )

    # y = 400, x = 300

    class params(object):
        pass

    # fixme: Параметры доски? не сходится что-то !!!! даже на разных досках
    # square_size = 50.0  # ???  шаг в mm? откуда узнать?
    # pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    # pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    # pattern_points *= square_size

    # print pattern_points

    # adsf

    # print pattern_points
    # by cols up done from left
    # x, y | x, y
    # (9, 6)  # per row, per col
    a, b, c, d = 0, 0, 300, 400 # Ok
    pattern_size = (11, 7)
    pattern_size = (11, 5)

    # a, b, c, d = 0, 0, 1200, 200  # NOT
    # pattern_size = (7, 5)
    # a, b, c, d = 450, 250, 600, 450
    # pattern_size = (3, 5)
    # a, b, c, d = 300, 150, 500, 800  # Ok
    # Похоже шахматку !!нужно покрутить!!

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
    for fn in img_names:
        fn = p + "/0000000000.png"
        if True:
            img = cv2.imread(fn, 0)
            h, w = img.shape[:2]
            img_cpy = np.copy(img)

            cv2.rectangle(img, (a, b), (c, d), (0, 127, 0), 3)

            img[:, :] = np.ones((h, w), dtype=np.uint8) * 0
            img[b:d, a:c] = img_cpy[b:d, a:c]
            print w, h, img.shape
            cv2.imwrite("/tmp/zones.png", img)

            # px, py = 104, 155
            # img = img[py:py+179, px:px+174]
            # img = img[3*h/4:, 400:w/2]

            # cv2.imshow("a", img)
            # cv2.waitKey(10000)

            h, w = img.shape[:2]
            # img = cv2.resize(img, (int(0.5 * w), int(0.5 * h)), interpolation=cv2.INTER_CUBIC)

        if img is None:
            print("Failed to load", fn)
            continue


        found, corners = cv2.findChessboardCorners(img, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ADAPTIVE_THRESH )

        # break
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        # if debug_dir:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)

        cv2.imwrite("/tmp/chess.png", vis)

        # cv2.imshow("a", vis)
        # cv2.waitKey(10000)
        # break
        #     path, name, ext = splitfn(fn)
        #     outfile = debug_dir + name + '_chess.png'
        #     cv2.imwrite(outfile, vis)
        #     if found:
        #         img_names_undistort.append(outfile)

        if not found:
            print 'chessboard not found'
            break
            continue

        # img_points.append(corners.reshape(-1, 2))
        # obj_points.append(pattern_points)

        print 'ok'
        break

    # # calculate camera distortion
    # rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    #
    # print("\nRMS:", rms)
    # print("camera matrix:\n", camera_matrix)
    # print("distortion coefficients: ", dist_coefs.ravel())

    # # undistort the image with the calibration
    # print('')
    # for img_found in img_names_undistort:
    #     img = cv2.imread(img_found)
    #
    #     h,  w = img.shape[:2]
    #     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    #
    #     dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    #
    #     # crop and save the image
    #     x, y, w, h = roi
    #     dst = dst[y:y+h, x:x+w]
    #     outfile = img_found + '_undistorted.png'
    #     print('Undistorted image written to: %s' % outfile)
    #     cv2.imwrite(outfile, dst)
    #
    # cv2.destroyAllWindows()
