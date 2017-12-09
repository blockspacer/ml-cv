# encoding: utf-8

import cv2

import numpy as np

gray_l = cv2.imread('../dataset-2014-bicycle1/im0.png', 0)

h, w = gray_l.shape[:2]

dist = None#np.array([0, 0, 0, 0, 0])
mtx = np.array([5299.313, 0, 1263.818, 0, 5299.313, 977.763, 0, 0, 1], dtype=np.float32)
mtx = mtx.reshape((3,3))
print mtx

newcameramtx = np.copy(mtx)
newcameramtx[0, 0] *= 0.5
newcameramtx[1, 1] *= 0.5
# newcameramtx[0, 2] *= 0.5
# newcameramtx[1, 2] *= 0.5

r = np.eye(3)

image_size = w, h
mapx,mapy = cv2.initUndistortRectifyMap(
	mtx,dist,r,newcameramtx, image_size, cv2.CV_16SC2)

print mapx.shape, mapy.shape

dst = cv2.remap(gray_l, mapx, mapy, cv2.INTER_LINEAR)

cv2.imwrite('/tmp/im0_dst.png', dst)