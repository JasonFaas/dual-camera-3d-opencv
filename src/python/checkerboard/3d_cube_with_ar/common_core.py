import numpy as np
import cv2 as cv

class CommonCore:

    def __init__(self, board_size):
        self.board_size = board_size


    def sort_points_for_extremes(self, corners2):
        if corners2.shape[1] == 1:
            x_min = np.min(corners2[:, 0, 0])
            y_min = np.min(corners2[:, 0, 1])
            x_max = np.max(corners2[:, 0, 0])
            y_max = np.max(corners2[:, 0, 1])
        else:
            x_min = np.min(corners2[0, :, 0])
            y_min = np.min(corners2[0, :, 1])
            x_max = np.max(corners2[0, :, 0])
            y_max = np.max(corners2[0, :, 1])
        return x_min, y_min, x_max, y_max

    def find_chessboard_corners_1(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # TODO improve this, there are a lot of false-negatives
        chessboard_corners_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 91, 0)
        ret_1, corners = cv.findChessboardCorners(chessboard_corners_img, self.board_size, corners=None,
                                                  flags=cv.CALIB_CB_FAST_CHECK)
        if not ret_1:
            chessboard_corners_img = gray
            ret_1, corners = cv.findChessboardCorners(chessboard_corners_img, self.board_size, corners=None)
        return chessboard_corners_img, corners, ret_1

    def find_chesboard_corners_2(self, chessboard_corners_img, corners, board_rotation):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(chessboard_corners_img, corners, (11, 11), (-1, -1), criteria)
        corners2 = board_rotation.get_rotated_corners(corners2)
        return corners2