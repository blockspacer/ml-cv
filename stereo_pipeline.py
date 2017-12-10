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

from sklearn import linear_model

# print cv2.__version__

from matplotlib import pyplot as plt

# https://github.com/utiasSTARS/pykitti
import sys

sys.path.append("pykitti_master")  # fixme: baaaad...
import pykitti

import scipy.optimize
import functools


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
        if norm(verts[i][0:3]) < 60:
            res.append(verts[i])

    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(res))).encode('utf-8'))
        np.savetxt(f, res, fmt='%f %f %f %d %d %d ')


###########################################################################

def get_correction_sim():
    Ry = np.matrix(np.eye(4))
    # Ry[0, 0] = 0
    # Ry[2, 2] = 0
    # Ry[0, 2] = 0
    # Ry[2, 0] = 0

    Rx = np.matrix(np.eye(4))
    theta = 0.25
    Rx[1, 1] = np.cos(theta)
    Rx[2, 2] = np.cos(theta)
    Rx[1, 2] = -np.sin(theta)
    Rx[2, 1] = np.sin(theta)

    R = Rx * Ry * np.matrix(np.eye(4))

    # R[2, 3] = 1.67

    return R


def get_correction_end():
    R = np.matrix(np.eye(4))

    Rr = np.matrix([[0.9999, -0.0058, 0.0105],
                    [-0.0058, 0.5404, 0.8414],
                    [-0.0105, -0.8414, 0.5403]])

    Rr = np.matrix([[0.9999, -0.0058, -0.0105],
                    [-0.0058, 0.5404, -0.8414],
                    [0.0105, 0.8414, 0.5403]])

    # R[0:3, 0:3] = Rr

    # [0.0199  1.5901  0.3883]
    R[0, 3] = -0.0199
    R[1, 3] = -1.5901
    R[2, 3] = -0.3883

    return R


def plane(x, y, params):
    # def plane(x, z, params):
    # print z, x
    a = params[0]
    b = params[1]
    c = params[2]
    z = a * x + b * y + c
    # y = a * x + b * z + c
    return z


def error(params, points):
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff ** 2
    return result


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def do_it():
    basedir = '/mnt/d1/datasets'
    date = '2011_09_26'
    drive = '0056'
    raw = True

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive, range(82, 83, 1))
    dataset.load_calib()
    c = dataset.calib
    dataset.load_gray(format='cv2')  # Loads images as uint8 grayscale

    # select
    idx = 0
    l = dataset.gray[idx].left
    r = dataset.gray[idx].right

    # configure
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
    # print help(stereo)
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

    # fixme: calc correction
    if not raw:
        Q = get_correction_end() * get_correction_sim() * np.matrix(Q)
    else:
        Q = get_correction_sim() * np.matrix(Q)

    points = cv2.reprojectImageTo3D(disp, Q)

    colors = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
    # mask = disp > disp.min()
    mask = np.zeros(points.shape, dtype=np.bool)
    mask[220:, 400: 800] = 1

    # plt.imshow(mask)
    # plt.show()


    out_points = points[mask]
    out_points_clone = np.copy(out_points)
    print out_points_clone.shape
    out_colors = colors[mask]
    if raw:
        # fit plain
        data_frame = out_points_clone.reshape(-1, 3)
        data_frame_new = []

        for r in data_frame:
            have_inf = False
            for p in r:
                if p == np.inf or abs(p) > 40:  # outliers rejection
                    have_inf = True

            if not have_inf:
                data_frame_new.append(r)

        data_frame_new = np.array(data_frame_new)

        X_train = data_frame_new[:, 0:2]
        y_train = data_frame_new[:, 2]
        print X_train

        # plt.plot(X_train[:, 0], y_train, 'o')
        # plt.plot(X_train[:, 1], y_train, 'o')
        # plt.axes().set_aspect('equal', 'datalim')
        # plt.grid()
        # plt.show()

        # Create linear regression object
        regr = linear_model.LinearRegression(fit_intercept=True)
        # regr = linear_model.RANSACRegressor()  # fit_intercept=True)

        # Train the model using the training sets
        # https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points
        regr.fit(X_train, y_train)
        # z = ax + by + c
        # -dz + ax + by = -c
        a = regr.coef_[0]
        b = regr.coef_[1]
        c = regr.intercept_
        d = -1  # z

        N = np.array([a, b, d])
        n = N / norm(N)
        n_coord = np.array([0, 0, 1])  # by z

        # T - ?
        # https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/1.-vectors-and-matrices/part-c-parametric-equations-for-curves/session-16-intersection-of-a-line-and-a-plane/MIT18_02SC_we_9_comb.pdf
        # http://www.ambrsoft.com/TrigoCalc/Plan3D/PlaneLineIntersection_.htm
        # x=at, y = bt, z=ct
        print n
        A, B, C, D = a, b, d, c
        a_, b_, c_ = n[0], n[1], n[2]
        x1, y1, z1 = 0, 0, 0

        x = x1 - a_ * (A * x1 + B * y1 + c_ * z1 + D) / (A * a_ + B * b_ + C * c_)
        y = y1 - b_ * (A * x1 + B * y1 + c_ * z1 + D) / (A * a_ + B * b_ + C * c_)
        z = z1 - c_ * (A * x1 + B * y1 + c_ * z1 + D) / (A * a_ + B * b_ + C * c_)

        out_points_ = np.array([x, y, z])
        # out_points_ = np.array([A, B, C])
        # out_points_ = np.array(n * 10)
        out_colors_ = np.array([255, 0, 0])
        print out_points_

        out_points = np.concatenate([out_points, out_points_])
        out_colors = np.concatenate([out_colors, out_colors_])

        # # fixme: не соптимизилось, из-за outliers, too long
        # fun = functools.partial(error, points=data_frame_new)
        # params0 = [0., 0., 0.]
        # res = scipy.optimize.minimize(fun, params0)
        # print params0, res

        # trouble
        # fixme: в датасете все в движении, как вычесть бэграунд?

        # R matrix
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        kk = np.cross(n_coord, n)
        k = kk / norm(kk)
        print k
        R = cv2.Rodrigues(k, None)
        print R[0]

    # Eval
    write_ply('out.ply', out_points, out_colors)


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
