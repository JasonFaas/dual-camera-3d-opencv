import numpy as np
import cv2 as cv

board_size = (7, 7)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_size[1]*board_size[0],3), np.float32)
objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Read image
fname = '../../resources/old_picture_hold/1/opencv_frame_1_0.png'
img = cv.imread(fname)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, board_size, None)

# If found, add object points, image points (after refining them)
if not ret:
    print('fail at ret')
    exit(0)

objpoints.append(objp)

corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
imgpoints.append(corners)

# Draw and display the corners
cv.drawChessboardCorners(img, board_size, corners, ret)


def show_image(image_to_draw):
    cv.imshow('img', image_to_draw)
    # print("pass_" + fname)
    while True:
        k = cv.waitKey(10)
        if k % 256 == 27:
            break


show_image(img)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print("mtx: " + str(mtx))
# print("dist: " + str(dist))
np.savez("0_distorition_info.txt", mtx)
print(mtx)

img = cv.imread(fname)
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# print("ncm: " + str(newcameramtx))
# print("roi: " + str(roi))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
# cv2.imwrite('calibresult.png',dst)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0][1+1].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[0][7+7].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[0][7+1+7+1].ravel()), (0,0,255), 5)
    return img
# print(corners[0])
# print('a')
# print(corners)
# print('b')
# print(imgpoints)
# print(imgpoints[0][0])
# print(imgpoints[0][1])
# print(imgpoints[0][2])
# print(imgpoints[0])

# #draw 2d axis points
# img = draw(img, corners, imgpoints)
# show_image(img)


#3d Time
criteria2 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp2 = np.zeros((board_size[1]*board_size[0],3), np.float32)
objp2[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
axis2 = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# Find the rotation and translation vectors.
ret2,rvecs2, tvecs2 = cv.solvePnP(objp2, corners2, mtx, dist)
# project 3D points to image plane
imgpts2, jac2 = cv.projectPoints(axis2, rvecs2, tvecs2, mtx, dist)

print('f')
print(corners2)
print('g')
print(imgpts2)
def draw2(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
img = draw2(img,corners2,imgpts2)

cv.imshow('img',img)
k = cv.waitKey(0) & 0xFF
if k == ord('s'):
    cv.imwrite(fname[:6]+'.png', img)



axis3 = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
ret2,rvecs2, tvecs2 = cv.solvePnP(objp2, corners2, mtx, dist)
imgpts3, jac3 = cv.projectPoints(axis3, rvecs2, tvecs2, mtx, dist)

def draw3(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

img = draw3(img,corners2,imgpts3)
cv.imshow('img',img)
k = cv.waitKey(0) & 0xFF
if k == ord('s'):
    cv.imwrite(fname[:6]+'.png', img)

cv.destroyAllWindows()