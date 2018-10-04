# Goal of program:
# Display 3d cube on checkerboard
# Increase size of cube based on input of 'ar' button on checkerboard

import numpy as np
import cv2 as cv
import datetime
from draw_cube import DrawCube
from ar_input import ArInput
from board_rotation import BoardRotation

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
video = 'opencv_video_2018-10-02_11:15:45.129696_0.mkv'


board_width = 7
board_size = (board_width, board_width)

draw_cube = DrawCube()
ar_input = ArInput(board_width, resources_path)
board_rotation = BoardRotation(0)


def analyze_image_and_add_ar(frame_to_analyze):
    img = frame_to_analyze
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners

    chessboard_corners_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 91, 0)
    ret_1, corners = cv.findChessboardCorners(chessboard_corners_img, board_size, corners=None, flags=cv.CALIB_CB_FAST_CHECK)

    # fail program if points not found
    if not ret_1:
        frame_wait_cnt = 10
        print('no checkerboard detected, wait %s frames' % str(frame_wait_cnt))
        cv.imshow('img', img)
        return frame_wait_cnt

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv.cornerSubPix(chessboard_corners_img, corners, (11, 11), (-1, -1), criteria)

    corners2 = board_rotation.get_rotated_corners(corners2)

    cube_size, img, rotate = ar_input.look_for_cube_size_v2(img, corners2)
    if rotate:
        board_rotation.update_rotation(corners2[6, 0])

    img = draw_cube.draw_cube_pieces(corners, corners2, img, board_size, cube_size=cube_size)

    cv.imshow('img', img)

    return 0

for picture in pictures:
    # Read image
    fname = '%s%s%s' % (resources_path, folder_path, picture)
    img_orig = cv.imread(fname)
    analyze_image_and_add_ar(img_orig)
    if cv.waitKey(0) & 0xFF == ord('q'):
        print("exit requested")
        cv.destroyAllWindows()
        exit(0)

vname = '%s%s%s' % (resources_path, folder_path, video)
cap = cv.VideoCapture(vname)
for i in range(250):
    cap.read()
while cap.isOpened():
    ret, img_orig = cap.read()
    if not ret:
        print("video is over")
        break
    frames_to_skip = analyze_image_and_add_ar(img_orig)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("exit requested")
        cv.destroyAllWindows()
        exit(0)

    # Skip frames when checkerboard is not found
    while cap.isOpened() and frames_to_skip > 0:
        ret, img_orig = cap.read()
        frames_to_skip -= 1

cv.destroyAllWindows()
