# Goal of script is to display a cube onto a picture of me holding a checkerboard
import numpy as np
import cv2 as cv
import datetime


def show_image(image_to_draw):
    cv.imshow('img', image_to_draw)
    cv.waitKey(0) & 0xFF

    datetime_now = str(datetime.datetime.now()).replace(' ', '_')
    img_name = "../../resources/saved/" + "opencv_frame_{}_{}.png".format(datetime_now, 'img')
    cv.imwrite(img_name, image_to_draw)


board_size = (7, 7)

# Read image
fname = '../../resources/old_picture_hold/1/opencv_frame_1_0.png'
img_orig = cv.imread(fname)
img = img_orig.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, board_size, None)
# fail program if points not found
if not ret:
    print('fail at ret')
    exit(0)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_size[1]*board_size[0],3), np.float32)
objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

objpoints.append(objp)

corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
imgpoints.append(corners)


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

#3d Time
criteria2 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp2 = np.zeros((board_size[1]*board_size[0],3), np.float32)
objp2[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
axis2 = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# Find the rotation and translation vectors.
ret2,rvecs2, tvecs2 = cv.solvePnP(objp2, corners2, mtx, dist)
# project 3D points to image plane
imgpts2, jac2 = cv.projectPoints(axis2, rvecs2, tvecs2, mtx, dist)


# TODO: reenable drawing axis lines
# def draw2(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#     img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#     img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#     return img
# img = draw2(img,corners2,imgpts2)
#
# show_image(img)


edge_size = 2
x_offset = 3
y_offset = 2
axis3 = np.float32([[x_offset, y_offset, 0],            [x_offset, y_offset + edge_size, 0],            [x_offset + edge_size, y_offset + edge_size, 0],            [x_offset + edge_size, y_offset, 0],
                    [x_offset, y_offset, -edge_size],   [x_offset, y_offset + edge_size, -edge_size],   [x_offset + edge_size, y_offset + edge_size, -edge_size],   [x_offset + edge_size, y_offset, -edge_size]])
ret2,rvecs2, tvecs2 = cv.solvePnP(objp2, corners2, mtx, dist)
imgpts3, jac3 = cv.projectPoints(axis3, rvecs2, tvecs2, mtx, dist)

def draw3(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw back left wall
    back_pts_2 = np.empty((0, 2), dtype=int)
    back_pts_2 = np.append(back_pts_2, [imgpts[0]], axis=0)
    back_pts_2 = np.append(back_pts_2, [imgpts[4]], axis=0)
    back_pts_2 = np.append(back_pts_2, [imgpts[5]], axis=0)
    back_pts_2 = np.append(back_pts_2, [imgpts[1]], axis=0)
    img = cv.drawContours(img, [back_pts_2],-1,(100,100,255),10)

    # draw back right wall
    back_pts_2 = np.empty((0, 2), dtype=int)
    back_pts_2 = np.append(back_pts_2, [imgpts[0]], axis=0)
    back_pts_2 = np.append(back_pts_2, [imgpts[4]], axis=0)
    back_pts_2 = np.append(back_pts_2, [imgpts[7]], axis=0)
    back_pts_2 = np.append(back_pts_2, [imgpts[3]], axis=0)
    img = cv.drawContours(img, [back_pts_2],-1,(255,100,100),-1)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-1)
    # draw pillars in blue color
    for i,j in zip(range(0,4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), -1)

    return img

img = draw3(img,imgpts3)
show_image(img)

cv.destroyAllWindows()
