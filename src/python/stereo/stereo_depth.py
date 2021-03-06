import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('opencv_frame_8_0.png',0)
imgR = cv2.imread('opencv_frame_8_1.png',0)

stereo = cv2.stereo.createStereoBM(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()


