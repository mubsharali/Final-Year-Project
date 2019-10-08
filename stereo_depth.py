import sys
import numpy as np
import cv2 as cv

REMAP_INTERPOLATION = cv.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 1024


print('loading images...')
imgL = cv.imread('D:/Images/left_mob.jpg')  # downscale images for faster processing
imgR = cv.imread('D:/Images/right_mob.jpg')
    #imgL = cv.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 21)
    #imgR = cv.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 21)
imgL = cv.GaussianBlur(imgL, (15, 15), 0)
imgR = cv.GaussianBlur(imgR, (15, 15), 0)
hl, wl = imgL.shape[:2]
hr, wr = imgR.shape[:2]
    # reading camera parameters after calibration
Rc = np.load('D:/Images/Scalib.npz')
    # undistorting images
    # newCameraMtxR, roiR = cv.getOptimalNewCameraMatrix(Rc['M2'], Rc['dist2'], (wr, hr), 1, (wr, hr))
    # udImgR = cv.undistort(imgR, Rc['M2'], Rc['dist2'], None, newCameraMtxR)  #undistorted Right Image
    # newCameraMtxL, roiL = cv.getOptimalNewCameraMatrix(Rc['M1'], Rc['dist1'], (wl, hl), 1, (wl, hl))
    # udImgL = cv.undistort(imgL, Rc['M1'], Rc['dist1'], None, newCameraMtxL)

rectify_scale = 0  # 0=full crop, 1=no crop
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(Rc["M1"], Rc["dist1"], Rc["M2"], Rc["dist2"], (640, 480), Rc["R"],Rc["T"], alpha=rectify_scale)
left_maps = cv.initUndistortRectifyMap(Rc["M1"], Rc["dist1"], R1, P1, (640, 480), cv.CV_16SC2)
right_maps = cv.initUndistortRectifyMap(Rc["M2"], Rc["dist2"], R2, P2, (640, 480), cv.CV_16SC2)
udImgL = cv.remap(imgL, left_maps[0], left_maps[1], cv.INTER_LANCZOS4)
udImgR = cv.remap(imgR, right_maps[0], right_maps[1], cv.INTER_LANCZOS4)

udImgR = cv.cvtColor(udImgR, cv.COLOR_BGR2GRAY)  # converting to grayScale
udImgL = cv.cvtColor(udImgL, cv.COLOR_BGR2GRAY)
# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images
stereoMatcher = cv.StereoBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setROI1(roi1)
stereoMatcher.setROI2(roi2)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)

# Grab both frames first, then retrieve to minimize latency between cameras
depth = stereoMatcher.compute(udImgL, udImgR)


cv.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
cv.waitKey()
cv.destroyAllWindows()