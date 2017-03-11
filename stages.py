# encoding: utf-8

### Global alg:
# http://aishack.in/tutorials/calibrating-undistorting-opencv-oh-yeah/
# http://stackoverflow.com/questions/15018620/findchessboardcorners-cannot-detect-chessboard-on-very-large-images-by-long-foca
# Looks good - http://docs.opencv.org/trunk/dc/dbb/tutorial_py_calibration.html
#
# Полный алгоритм
# http://moluch.ru/archive/118/32662/

# Kinnlect
# https://habrahabr.ru/post/272629/

# Step 0: On cam and stereo pair calibrate - Очень важно !!!
# FOR START
#
# Q:Как и какую ошибк минимизирвоать? Как посчитать?
# "Please make sure you have sufficient number of images at different
# depth form camera and at different
# orientation that will lead to less re projection error"
# A: cv2.calibrateCamera - return reproj. error, Но тоже недостаточно
# http://stackoverflow.com/questions/11918315/does-a-smaller-reprojection-error-always-means-better-calibration
#
# Q: Что на зависит от установки пары? K, K', F, R'(?) T', F, E
# A: а разве еще что-то нужно из матриц?
#
# Good advice: http://stackoverflow.com/questions/12794876/how-to-verify-the-correctness-of-calibration-of-a-webcam
# Good advice: http://stackoverflow.com/questions/24130884/opencv-stereo-camera-calibration-image-rectification
# "Now, about the stereo calibration itself.
# Best way that I found to achieve a good calibration is to
# separately calibrate each camera intrinsics (using the calibrateCamera
# function) then the extrinsics (using stereoCalibrate) using the
#  intrinsics as a guess. "
# fixme: но будет не так, скорее всего будет две отдельно откалибр.(K, R, T...) надеюсь в одной системе коорд. где-то
# и из них нужно будет вычислить R, T, F степео пары
# Q: Хотяяя... А почему не откалибровать все на заводе, координаты будут в левой камере

# Step 1: ???

# Step 2: ???

####



# All
# http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo

# After rectification
# Downl. templ
# Only rectified matrix(?) images to(?)
# wget http://vision.middlebury.edu/stereo/data/scenes2014/datasets/Bicycle1-perfect/calib.txt
# wget http://vision.middlebury.edu/stereo/data/scenes2014/datasets/Bicycle1-perfect/im0.png
# wget http://vision.middlebury.edu/stereo/data/scenes2014/datasets/Bicycle1-perfect/im1.png


# After + before
# MAIN: http://www.cvlibs.net/datasets/kitti/raw_data.php

# After rectification
# !!! Good one
# http://www.cvlibs.net/datasets/karlsruhe_sequences/

# Troubles:
# Разная яркость картинки - автоматы

import cv2
import numpy as np
import numpy

from numpy.linalg import norm

print cv2.__version__

from matplotlib import pyplot as plt

# https://github.com/utiasSTARS/pykitti
import sys

sys.path.append("pykitti_master")  # fixme: baaaad...
import pykitti


def load_params(fn):
    params_str = None
    with open(fn) as f:
        params_str = f.readlines()

    params = {}
    for param in params_str:
        param = param.strip()
        pair = param.split("=")
        key = pair[0]
        if "cam" in key:
            # fixme: Alarm - it's for rectified view !!!
            # http://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/
            # [f 0 cx; 0 f cy; 0 0 1]
            # K
            rows = pair[1].replace(']', '').replace('[', '').split(';')
            P = []
            for row in rows:
                P.append(numpy.fromstring(row.strip(), sep=" "))

            value = np.array(P)
            print int(value[0][2])
        else:
            value = float(pair[1])

        params[key] = value
        print key, ":"
        print value

    return params


def get_Kl(params):
    return params['cam0']


def get_Kr(params):
    return params['cam1']


def find_stereopairs():
    # Calibrate both cameras
    # K, R, t / K', R' t'

    # Both image sync, undistort and rectification

    # !! Triangulation
    # Find pairs on rectif. images

    # (?) come back to found real X,

    # fixme: Как перейти к 3D точкам?

    # fixme: disparity map

    # Make depth map ??

    pass


# fixme: может сразу сделать стерео калибровку
# http://stackoverflow.com/questions/27431062/stereocalibration-in-opencv-on-python

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

    res = []

    for i in range(len(verts)):
        if norm(verts[i][0:3]) < 30:
            res.append(verts[i])

    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(res))).encode('utf-8'))
        np.savetxt(f, res, fmt='%f %f %f %d %d %d ')


def get_correction():
    Ry = np.matrix(np.eye(4))
    Ry[0, 0] = 0
    Ry[2, 2] = 0
    Ry[0, 2] = 1
    Ry[2, 0] = -1

    Rx = np.matrix(np.eye(4))
    Rx[1, 1] = 0
    Rx[2, 2] = 0
    Rx[1, 2] = 1
    Rx[2, 1] = -1

    R = Rx * Ry * np.matrix(np.eye(4))

    R[2, 3] = 1.67

    return R


def do_it():
    basedir = '/mnt/d1/datasets'
    date = '2011_09_26'
    drive = '0056'

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive, range(75, 90, 1))
    dataset.load_calib()
    c = dataset.calib
    dataset.load_gray(format='cv2')  # Loads images as uint8 grayscale

    # select
    idx = 7
    l = dataset.gray[idx].left
    r = dataset.gray[idx].right

    # configure
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
    print help(stereo)
    stereo.setUniquenessRatio(20)
    stereo.setTextureThreshold(100)

    # run
    disp = stereo.compute(l, r)
    disp = np.array(disp, dtype=np.float32) / 16.

    # to 3d
    cx = c.P_rect_00[0, 2]
    cy = c.P_rect_00[1, 2]
    f = c.P_rect_00[0, 0]
    cx_r = c.P_rect_01[0, 2]
    Tx = c.T_01[0]
    Q = np.array([[1, 0, 0, - cx],
                  [0, 1, 0, - cy],
                  [0, 0, 0, f],
                  [0, 0, -1. / Tx, (cx - cx_r) / Tx]])

    Q = get_correction() * np.matrix(Q)

    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    write_ply('out.ply', out_points, out_colors)

    # trouble
    # fixme: в датасете все в движении, как вычесть бэграунд?


def do_it1():
    # Parse cam params
    # http://wiki.ros.org/kinect_calibration/technical
    root = "dataset-2014-bicycle1/"
    fn = root + "calib.txt"
    params = load_params(fn)

    # fixme: почему cx1 != cx0, baseline != 0
    # fixme: как из этих матриц получить что-то?
    print get_Kl(params) - get_Kr(params)

    X, Y, Z = 0., 0, 1
    M = np.array([[X, Y, Z, 1.]]).T
    P0 = get_Kl(params)
    K = np.c_[P0, np.zeros(3)]
    print K
    sm = np.dot(K, M)
    UV = sm / sm[2][0]
    print np.array(UV, np.int32)

    # Task0: Move camera center?

    # TaskN: rectification

    # Load images
    img = cv2.imread(root + 'im0.png')  # , 0)
    cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

    # Draw
    small = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    # cv2.imshow('image', small)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Article:
    # http://ece631web.groups.et.byu.net/Lectures/ECEn631%2014%20-%20Calibration%20and%20Rectification.pdf
    #
    # T = Ol-Or


if __name__ == '__main__':
    """
        A*[R|t], K - rotation
    K = A ?
    [R|t] - extrinsic params
    A - intris.
    R - rotation-translation

    Need Intcisics/Intr Params BOTH!!
    "Let P be a camera matrix representing a general projective camera.
    We wish to find the
    camera centre, the orientation of the camera and the internal parameters of the camera
    from P."

    P = A*[R|t] - camera projection matrix
    x = P*X

    P = K[I|0]  P'= K' [R'|t']
    Test dataset P = K[I|0], P'= K'[I|t]

    The epipolar geometry is represented by a 3 × 3
    matrix called the fundamental matrix F.

    K - camera calibr. matrix
    K -> K + R -> K(internal) + (tilda_C + R)(external) -> t = -R * tilda_C
    x = KR[I|-tilda_C]X

    F - for epipolar

    COORDS:
    - camera coordinate frame
    - world coordinate frame

    How split P?
    decomposeProjectionMatrix

    """

    np.set_printoptions(precision=4, suppress=True)

    if True:
        do_it()

    if False:
        do_it1()
