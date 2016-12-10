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

def do_it():
    basedir = '/home/zaqwes/tools/datasets'
    date = '2011_09_26'
    drive = '0052'

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive, range(0, 50, 5))

    # Data are loaded only if requested
    dataset.load_calib()
    print dataset.calib.P_rect_00
    print dataset.calib.P_rect_01

    # Calc back, to real coord system
    print dataset.calib.T_01
    # print dataset.P_rect_01

    dataset.load_gray(format='cv2')  # Loads images as uint8 grayscale
    # dataset.load_rgb(format='cv2')  # Loads images as uint8 with BGR ordering

    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
    disp_gray = stereo.compute(dataset.gray[0].left, dataset.gray[0].right)

    # fixme: что лежит по этим индексам
    # это номер стереопары изображений - 0-1, 2-3 камеры
    #print dataset.gray[1]

    # plt.imshow(disp_gray)
    # plt.show()

    # How calc Q?

    # dataset.gray


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
