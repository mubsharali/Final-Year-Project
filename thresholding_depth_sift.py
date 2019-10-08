#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import win32com.client as wincl

from matplotlib import pyplot as plt
import glob

if __name__ == '__main__':

    dis = []
    print('loading images...')
    imgL = cv.imread('D:/Images/left_image_5.jpg')  # downscale images for faster processing
    imgR = cv.imread('D:/Images/right_image_5.jpg')
    # cv.imshow('Rimg', imgR);
    # imgL = cv.bilateralFilter(imgL, 12, 75, 75)
    # imgR = cv.bilateralFilter(imgR, 12, 75, 75)
    imgL = cv.GaussianBlur(imgL, (35, 35), 0)
    imgR = cv.GaussianBlur(imgR, (35, 35), 0)
    hl, wl = imgL.shape[:2]
    hr, wr = imgR.shape[:2]
    # imgL.show();

    # reading camera parameters after calibration
    Rc = np.load('D:/Images/Scalib.npz')
    # undistorting images
    # newCameraMtxR, roiR = cv.getOptimalNewCameraMatrix(Rc['M2'], Rc['dist2'], (wr, hr), 1, (wr, hr))
    # udImgR = cv.undistort(imgR, Rc['M2'], Rc['dist2'], None, newCameraMtxR)  # undistorted Right Image
    # newCameraMtxL, roiL = cv.getOptimalNewCameraMatrix(Rc['M1'], Rc['dist1'], (wl, hl), 1, (wl, hl))
    # udImgL = cv.undistort(imgL, Rc['M1'], Rc['dist1'], None, newCameraMtxL)

    rectify_scale = 0  # 0=full crop, 1=no crop
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(Rc["M1"], Rc["dist1"], Rc["M2"], Rc["dist2"], (640, 480), Rc["R"],
                                                     Rc["T"], alpha=rectify_scale)
    left_maps = cv.initUndistortRectifyMap(Rc["M1"], Rc["dist1"], R1, P1, (640, 480), cv.CV_16SC2)
    right_maps = cv.initUndistortRectifyMap(Rc["M2"], Rc["dist2"], R2, P2, (640, 480), cv.CV_16SC2)
    udImgL = cv.remap(imgL, left_maps[0], left_maps[1], cv.INTER_LANCZOS4)
    udImgR = cv.remap(imgR, right_maps[0], right_maps[1], cv.INTER_LANCZOS4)

    udImgR = cv.cvtColor(udImgR, cv.COLOR_BGR2GRAY)  # converting to grayScale
    udImgL = cv.cvtColor(udImgL, cv.COLOR_BGR2GRAY)
    good = []
    pts1 = []
    pts2 = []
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT

    kp1, des1 = sift.detectAndCompute(udImgL, None)
    kp2, des2 = sift.detectAndCompute(udImgR, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=30)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    dis = [];
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            left_pt = kp1[m.queryIdx].pt  # left_pt is (x,y)
            right_pt = kp2[m.trainIdx].pt
            dispartity = abs(left_pt[0] - right_pt[0])  # left_pt[0] means x direction i.e pixel column value
            dis.append(dispartity)
            z = 948 * 0.0635 / dispartity
            #distance.append(z)
            # print('distance: ',z)
            # print('Point: ', abs(left_pt[0]))
    # print('End loading images...')
    dis = (1.0/dis)
    distance = (948 * 0.0635*dis)
    distance = distance.sort()
    min_distance = 0
    print("Distance Array", distance);
    '''for i in range(distance.__len__()):
        threshhold_values = (distance[i] - 0.5 > distance & distance < distance[i] + 0.5)
        if threshhold_values.__len__() >= 3:
            min_distance = distance[i]
            break
'''

    #z = 948 * 0.0635 / max(dis)
    print('Object DISTANCE: ', min_distance)
    #print('Point: ', left_pt)

    msg = 'Obstacle distance ' + str("{:.2f}".format(z)) + ' meter'

    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Speak(msg)
