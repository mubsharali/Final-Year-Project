import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('D:/Images/left_img.jpg',0)
imgR = cv.imread('D:/Images/right_img.jpg',0)

hl, wl = imgL.shape[:2]
hr, wr = imgR.shape[:2]
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

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(udImgL,udImgR)
plt.imshow(disparity,'gray')
plt.show()