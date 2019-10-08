#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import glob

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
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':
    print('loading images...')
    imgL = cv.imread('D:/Images/left_mob.jpg')  # downscale images for faster processing
    imgR = cv.imread('D:/Images/right_mob.jpg')
    hl, wl = imgL.shape[:2]
    hr, wr = imgR.shape[:2]
    Rc = np.load('D:/Images/Scalib.npz')
    # undistorting images
    newCameraMtxR, roiR = cv.getOptimalNewCameraMatrix(Rc['M2'], Rc['dist2'], (wr, hr), 1, (wr, hr))
    udImgR = cv.undistort(imgR, Rc['M2'], Rc['dist2'], None, newCameraMtxR)  # undistorted Right Image
    newCameraMtxL, roiL = cv.getOptimalNewCameraMatrix(Rc['M1'], Rc['dist1'], (wl, hl), 1, (wl, hl))
    udImgL = cv.undistort(imgL, Rc['M1'], Rc['dist1'], None, newCameraMtxL)
    # disparity range is tuned for 'aloe' image pair
    rectify_scale = 0  # 0=full crop, 1=no crop
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(Rc["M1"], Rc["dist1"], Rc["M2"], Rc["dist2"], (640, 480), Rc["R"],
                                                     Rc["T"], alpha=rectify_scale)
    left_maps = cv.initUndistortRectifyMap(Rc["M1"], Rc["dist1"], R1, P1, (640, 480), cv.CV_16SC2)
    right_maps = cv.initUndistortRectifyMap(Rc["M2"], Rc["dist2"], R2, P2, (640, 480), cv.CV_16SC2)
    udImgL = cv.remap(udImgL, left_maps[0], left_maps[1], cv.INTER_LANCZOS4)
    udImgR = cv.remap(udImgR, right_maps[0], right_maps[1], cv.INTER_LANCZOS4)
    cv.imshow('leftImg', udImgL)
    cv.imshow('rightImg', udImgR)

    cv.waitKey()
    cv.destroyAllWindows()
