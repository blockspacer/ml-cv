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

# http://aishack.in/tutorials/calibrating-undistorting-opencv-oh-yeah/
# http://stackoverflow.com/questions/15018620/findchessboardcorners-cannot-detect-chessboard-on-very-large-images-by-long-foca

if __name__ == '__main__':
    p = '/home/zaqwes/datasets/2011_09_26/2011_09_26_drive_0119_extract/image_00/data'
    img_names = glob( p+"/*.png" )

    # y = 400, x = 300

    pattern_size = (11, 7)#(9, 6)  # per row, per col

    # fixme: Параметры доски? не сходится что-то !!!! даже на разных досках
    square_size = 1.0  # ???
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # print pattern_points

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
    for fn in img_names:
        # print('processing %s... ' % fn, end='')
        fn = '/tmp/2Bfo4.png'
        fn = p + "/0000000000.png"

        if True:
            img = cv2.imread(fn, 0)
            px, py = 104, 155
            # img = img[py:py+179, px:px+174]

            # cv2.imshow("a", img)
            # cv2.waitKey(10000)

            h, w = img.shape[:2]
            # img = cv2.resize(img, (int(0.5 * w), int(0.5 * h)), interpolation=cv2.INTER_CUBIC)

        if img is None:
            print("Failed to load", fn)
            continue


        print h, w
        found, corners = cv2.findChessboardCorners(img, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ADAPTIVE_THRESH )
        print found

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
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        print 'ok'

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

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
