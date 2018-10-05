# Goal of program:
# Display 3d cube on checkerboard
# Increase size of cube based on input of 'ar' button on checkerboard

import numpy as np
import cv2 as cv
import datetime
from draw_cube import DrawCube
from ar_input import ArInput
from board_rotation import BoardRotation
from common_core import CommonCore

resources_path = '../../resources/'
folder_path = 'checkboard_cube_with_ar_input/'
# current pictures descriptions:
# 0 should work fine
# 1 should work fine
# 2 should work and disagree with 3
# 3 should work and disagree with 2
# 4 should work and disagree with 5
# 5 should work and disagree with 4
# 6 should work and agree with 7
# 7 should work and agree with 6
pictures = [['opencv_frame_0_0.png', 'opencv_frame_0_1.png'], ['opencv_frame_7_0.png', 'opencv_frame_7_1.png'], ['opencv_frame_7_1.png', 'opencv_frame_7_0.png'], ['opencv_frame_5_0.png', 'opencv_frame_5_1.png']]


board_width = 7
board_size = (board_width, board_width)

draw_cube = DrawCube(resources_path)
common_core = CommonCore(board_size)
ar_input = ArInput(board_width, resources_path, '')
board_rotation_f1 = BoardRotation()
board_rotation_f2 = BoardRotation()


def analyze_image_and_add_ar(frame_1, frame_2):
    # frame_1 is suggestion and display frame, frame_2 is confirm frame
    img_1 = frame_1
    chessboard_img_1, corners1_1, ret_1 = common_core.find_chessboard_corners_1(img_1)

    # fail program if points not found
    if not ret_1:
        frame_wait_cnt = 10
        print('1st checkerboard NOT detected, wait %s frames' % str(frame_wait_cnt))
        cv.imshow('img', img_1)
        return frame_wait_cnt

    corners2_1 = common_core.find_chesboard_corners_2(chessboard_img_1, corners1_1, board_rotation_f1)
    chessboard_img_2, corners1_2, ret_2 = common_core.find_chessboard_corners_1(frame_2)
    if ret_2:
        corners2_2 = common_core.find_chesboard_corners_2(chessboard_img_2, corners1_2, board_rotation_f2)
    else:
        frame_wait_cnt = 10
        print('2nd checkerboard NOT detected, wait %s frames' % str(frame_wait_cnt))
        cv.imshow('img', img_1)
        return frame_wait_cnt

    # get button pressed info for 1, and verify if need
    img_1, button_pressed_1, roi_corners_1, roi_mask_1 = ar_input.look_for_cube_size_v2(img_1, corners2_1, draw_buttons=True)
    frame_2_result = frame_2
    if button_pressed_1 != -1:
        print("img_1 pressed:" + str(button_pressed_1))
        # frame_2 confirmation
        frame_2_result, button_pressed_2, _, _ = ar_input.look_for_cube_size_v2(frame_2, corners2_2, draw_buttons=True)
        print("\timg_2 pressed:" + str(button_pressed_2))
        if button_pressed_2 == button_pressed_1:
            img_1 = ar_input.place_inverted_button_on_checkerboard(img_1,
                                                                   button_pressed_1 - 1,
                                                                   roi_mask_1,
                                                                   roi_corners_1)
            if button_pressed_1 == 6:
                board_rotation_f1.update_rotation(corners2_1[6, 0])
                board_rotation_f2.update_rotation(corners2_2[6, 0])
            else:
                ar_input.update_cube_size(button_pressed_1)



    img = draw_cube.draw_cube_pieces(corners1_1,
                                     corners2_1,
                                     img_1,
                                     board_size,
                                     cube_size=ar_input.get_cube_size())

    cv.imshow('frame_2_result', frame_2_result)
    cv.imshow('img', img)

    return 0





pictures = []
for picture_group in pictures:
    # Read image
    fname_1 = '%s%s%s' % (resources_path, folder_path, picture_group[0])
    fname_2 = '%s%s%s' % (resources_path, folder_path, picture_group[1])
    img_orig_1 = cv.imread(fname_1)
    img_orig_2 = cv.imread(fname_2)
    analyze_image_and_add_ar(img_orig_1, img_orig_2)
    if cv.waitKey(0) & 0xFF == ord('q'):
        print("exit requested")
        cv.destroyAllWindows()
        exit(0)

# vname = '%s%s%s' % (resources_path, folder_path, video)
# cap = cv.VideoCapture(vname)
# for i in range(250):
#     cap.read()
# while cap.isOpened():
#     ret, img_orig = cap.read()
#     if not ret:
#         print("video is over")
#         break
#     frames_to_skip = analyze_image_and_add_ar(img_orig)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         print("exit requested")
#         cv.destroyAllWindows()
#         exit(0)
#
#     # Skip frames when checkerboard is not found
#     while cap.isOpened() and frames_to_skip > 0:
#         ret, img_orig = cap.read()
#         frames_to_skip -= 1



print("Hotkeys:")
quit = 'q'
screenshot = 's'
start_recording = 'r'
stop_recording = 't'
print("\t" + 'quit:' + quit)
print("\t" + 'screenshot:' + screenshot)
print("\t" + 'start_recording:' + start_recording)
print("\t" + 'stop_recording:' + stop_recording)

cam0 = cv.VideoCapture(0)
cam1 = cv.VideoCapture(1)


while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    # cv.imshow("Zero Test", frame0)
    # cv.imshow("One Test", frame1)

    if not ret0 or not ret1:
        break

    analyze_image_and_add_ar(frame0, frame1)

    key_press = cv.waitKey(1) & 0xFF
    if key_press == ord(quit):
        print("Escape hit, closing...")
        break

cv.destroyAllWindows()
