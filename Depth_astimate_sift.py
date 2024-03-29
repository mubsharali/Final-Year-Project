#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

if __name__ == '__main__':

    dis=[]
    print('loading images...')
    imgL = cv.imread('D:/Images/left_image_4.jpg')  # downscale images for faster processing
    imgR = cv.imread('D:/Images/right_image_4.jpg')
    #cv.imshow('Rimg', imgR);
    #imgL = cv.bilateralFilter(imgL, 12, 75, 75)
    #imgR = cv.bilateralFilter(imgR, 12, 75, 75)
    imgL = cv.GaussianBlur(imgL, (35, 35), 0)
    imgR = cv.GaussianBlur(imgR, (35, 35), 0)
    hl, wl = imgL.shape[:2]
    hr, wr = imgR.shape[:2]
    #imgL.show();

    # reading camera parameters after calibration
    Rc = np.load('D:/Images/Scalib.npz')
    # undistorting images
    newCameraMtxR, roiR = cv.getOptimalNewCameraMatrix(Rc['M2'], Rc['dist2'], (wr, hr), 1, (wr, hr))
    udImgR = cv.undistort(imgR, Rc['M2'], Rc['dist2'], None, newCameraMtxR)  #undistorted Right Image
    newCameraMtxL, roiL = cv.getOptimalNewCameraMatrix(Rc['M1'], Rc['dist1'], (wl, hl), 1, (wl, hl))
    udImgL = cv.undistort(imgL, Rc['M1'], Rc['dist1'], None, newCameraMtxL)

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

    dis=[];
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            left_pt = kp1[m.queryIdx].pt  #left_pt is (x,y)
            right_pt = kp2[m.trainIdx].pt
            dispartity = abs(left_pt[0] - right_pt[0]) #left_pt[0] means x direction i.e pixel column value
            dis.append(dispartity);
            z = 948 * 0.0635 / dispartity
            print('distance: ',z)
            print('Point: ', abs(left_pt[0]))
    #print('End loading images...')

    z = 948 * 0.0635 / max(dis)
    print('Object DISTANCE: ',z)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]


    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img1.shape
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2


    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(udImgL, udImgR, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(udImgR, udImgL, lines2, pts2, pts1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()


