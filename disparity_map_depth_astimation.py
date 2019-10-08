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
    #imgL = cv.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 21)
    #imgR = cv.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 21)
    imgL = cv.GaussianBlur(imgL, (25, 25), 0)
    imgR = cv.GaussianBlur(imgR, (25, 25), 0)
    hl, wl = imgL.shape[:2]
    hr, wr = imgR.shape[:2]
    # reading camera parameters after calibration
    Rc = np.load('D:/Images/Scalib.npz')
    # undistorting images
    # newCameraMtxR, roiR = cv.getOptimalNewCameraMatrix(Rc['M2'], Rc['dist2'], (wr, hr), 1, (wr, hr))
    # udImgR = cv.undistort(imgR, Rc['M2'], Rc ['dist2'], None, newCameraMtxR)  #undistorted Right Image
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
    window_size = 7
    min_disp = 32
    num_disp = 128 - min_disp
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=15,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=3,
                                  speckleWindowSize=45,
                                  speckleRange=16
                                  )

    print('computing disparity...')
    disp = stereo.compute(udImgL, udImgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...', disp[0][0])
    h, w = udImgL.shape[:2]

    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -948],  # 948 is focal length of cameras from camera matrix,after calibration
                    [0, 0, 1, 0]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()  # boolean mask of 2D after condition,where disparity is greater then minimum disparity
    out_points = points[mask]  # contain only those points where this above condition match
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)

    print('%s saved' % 'out.ply')

    cv.imshow('left', udImgL)
    cv.imshow('disparity', (disp - min_disp) / num_disp)

    b = 0.0635  # baseline
    D = b * 948 / disp  # depth for each point

    print('Max dasparity: ', disp.max())
    mat = np.matrix(D)
    print('distance: ', mat.min())
    with open('outfile.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

    #plt.subplot(121), plt.imshow((disp - min_disp) / num_disp)
    #plt.show()
    cv.waitKey()
    cv.destroyAllWindows()
