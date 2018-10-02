# Goal of program:
# Display 3d cube on checkerboard
# Increase size of cube based on input of 'ar' button on checkerboard

import numpy as np
import cv2 as cv
import datetime
from draw_cube import DrawCube

resources_path = '../../resources/'
folder_path = 'checkboard_cube_with_ar_input/'
# current pictures descriptions:
# 0 should work fine
# 1 should work fine
# 2 should work and disagree with 3
# 3 should work and disagree with 2
# 4 should work and agree with 5
# 5 should work and agree with 4
pictures = ['opencv_frame_0_0.png', 'opencv_frame_0_1.png', 'opencv_frame_7_0.png', 'opencv_frame_7_1.png', 'opencv_frame_5_0.png', 'opencv_frame_5_1.png']

board_size = (7, 7)

draw_cube = DrawCube()


def show_image(image_to_draw):
    cv.imshow('img', image_to_draw)
    cv.waitKey(0) & 0xFF

    datetime_now = str(datetime.datetime.now()).replace(' ', '_')
    img_name = resources_path + "saved/" + "opencv_frame_{}_{}.png".format(datetime_now, 'img')
    cv.imwrite(img_name, image_to_draw)


def analyze_image_and_add_ar(frame_to_analyze):
    # Read image
    fname = '%s%s%s' % (resources_path, folder_path, frame_to_analyze)
    img_orig = cv.imread(fname)
    img = img_orig.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, board_size, None)
    # fail program if points not found
    if not ret:
        print('fail at ret')
        show_image(img)
        return
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners)
    _, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # 3d Time
    objp2 = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    objp2[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    edge_size = 3
    x_offset = 0
    y_offset = 0
    z_offset = -0
    axis3 = np.float32([[x_offset, y_offset, z_offset], [x_offset, y_offset + edge_size, z_offset],
                        [x_offset + edge_size, y_offset + edge_size, z_offset],
                        [x_offset + edge_size, y_offset, z_offset],
                        [x_offset, y_offset, z_offset - edge_size],
                        [x_offset, y_offset + edge_size, z_offset - edge_size],
                        [x_offset + edge_size, y_offset + edge_size, z_offset - edge_size],
                        [x_offset + edge_size, y_offset, z_offset - edge_size]])
    _, rvecs2, tvecs2 = cv.solvePnP(objp2, corners2, mtx, dist)
    imgpts3, jac3 = cv.projectPoints(axis3, rvecs2, tvecs2, mtx, dist)
    # Read image to project and put it on image with same size as image to project to
    develjpg = cv.imread(resources_path + "to_project/devel_square.jpg")
    devel_jpg_full_size = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    dev_rows, dev_cols, _ = develjpg.shape
    devel_jpg_full_size[0:dev_rows, 0:dev_cols] = develjpg.copy()


    img = draw_cube.add_image_to_base(img, devel_jpg_full_size, np.array([imgpts3[0], imgpts3[3], imgpts3[2], imgpts3[1]]), dev_rows, dev_cols)


    img = draw_cube.draw_cube(img, imgpts3)
    show_image(img)


for picture in pictures:
    analyze_image_and_add_ar(picture)

cv.destroyAllWindows()
