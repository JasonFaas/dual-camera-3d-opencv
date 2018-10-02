# Goal of program:
# Display 3d cube on checkerboard
# Increase size of cube based on input of 'ar' button on checkerboard

import numpy as np
import cv2 as cv
import datetime
from draw_cube import DrawCube
from ar_input import ArInput

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

board_width = 7
board_size = (board_width, board_width)

draw_cube = DrawCube()

ar_input = ArInput(board_width)


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

    cube_size = ar_input.look_for_cube_size_v2(img, gray, corners)

    img = draw_cube.draw_cube_pieces(corners, gray, img, board_size, cube_size=cube_size)

    show_image(img)

for picture in pictures:
    analyze_image_and_add_ar(picture)

cv.destroyAllWindows()
