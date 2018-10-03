import cv2 as cv
import numpy as np
from operator import itemgetter
from draw_cube import DrawCube


class ArInput:

    def __init__(self, board_width, resource_path):
        self.board_width = board_width
        self.cube_size = 1
        self.resource_path = resource_path
        self.font = cv.imread("3d_cube_with_ar/font/white_rabbit_numbers.jpg")
        font_six_points = [[16, 20], [80, 80]]
        blue_tint = np.empty(self.font.shape, dtype=np.uint8)
        blue_tint[:,:] = (155, 155, 0)
        self.font[16: 16 + 80, 20: 20 + 80] = cv.subtract(self.font[16: 16 + 80, 20: 20 + 80], blue_tint[16: 16 + 80, 20: 20 + 80])
        self.font_inv = cv.bitwise_not(self.font)
        font_one_points = [[16, 155], [80, 80]]
        font_two_points = [[20, 155+80+50], [80, 80]]
        font_three_points = [[20, 155+(80+50) * 2], [80, 80]]
        font_four_points = [[22, 155+(80+50) * 3 - 5], [80, 80]]
        font_five_points = [[22, 155+(80+50) * 4 - 10], [80, 80]]
        self.font_point = np.array([font_one_points,
                                   font_two_points,
                                   font_three_points,
                                   font_four_points,
                                   font_five_points,
                                   font_six_points])
        self.draw_cube = DrawCube()

    def look_for_cube_size_v1(self, img, corners):
        cube_return = self.cube_size
        self.cube_size = (self.cube_size % 6) + 1
        return cube_return

    def look_for_cube_size_v2(self, img, corners2):

        # TODO put this into a verification
        x_min, y_min, x_max, y_max = self.sort_points_for_extremes(corners2)
        # print(smallest_x)
        left_most_point = corners2[self.board_width ** 2 - self.board_width, 0]
        # print(left_most_point)
        # print(smallest_y)
        top_most_point = corners2[0, 0]
        # print(top_most_point)
        # print(largest_x)
        right_most_point = corners2[self.board_width - 1, 0]
        # print(right_most_point)
        # print(largest_y)
        bottom_most_point = corners2[self.board_width ** 2 - 1, 0]
        # print(bottom_most_point)
        # print(top_most_point.reshape(2,-1).shape)
        # cv.line(img, top_most_point.reshape(2,-1), right_most_point.reshape(2,-1), (255, 0, 0), thickness=5)

        hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
        image_sums = np.zeros((6, 2), dtype=np.int32)

        greatest_sum = 0
        greatest_sum_roi_corners = -1
        for square in range(6):
            # acquire points from corners to use for ar buttons
            first_bad_format = corners2[self.board_width * square, 0]
            second_bad_format = corners2[self.board_width * square + 1, 0]
            third_bad_format = corners2[self.board_width * (square + 1) + 1, 0]
            fourth_bad_format = corners2[self.board_width * (square + 1), 0]
            first = (first_bad_format[0], first_bad_format[1])
            fourth = (fourth_bad_format[0], fourth_bad_format[1])
            second = (int(first[0] * 2 - second_bad_format[0]), int(first[1] * 2 - second_bad_format[1]))
            third = (int(fourth[0] * 2 - third_bad_format[0]), int(fourth[1] * 2 - third_bad_format[1]))

            # put points into arrays for use below
            roi_corners_for_mask = np.array([[first, second, third, fourth]], dtype=np.int32)
            roi_corners = np.array([[third], [second], [first], [fourth]], dtype=np.int32)

            # create mask with polygon
            mask = np.zeros(sat.shape, dtype=np.uint8)
            # channel_count = sat.shape[2]
            channel_count = 1
            ignore_mask_color = (255,) * channel_count
            cv.fillPoly(mask, roi_corners_for_mask, ignore_mask_color)

            # smooth image
            x_min, y_min, x_max, y_max = self.sort_points_for_extremes(roi_corners_for_mask)
            sat[y_min:y_max, x_min:x_max] = cv.medianBlur(sat[y_min:y_max, x_min:x_max], 9)
            val[y_min:y_max, x_min:x_max] = cv.medianBlur(val[y_min:y_max, x_min:x_max], 9)

            # TODO use ROI here for optimization
            # black out all area outside roi_corners
            masked_val_img = cv.bitwise_and(val, mask)
            masked_sat_img = cv.bitwise_and(sat, mask)

            # TODO use ROI here for optimization
            # threshold for high val and high sat
            _, masked_val_img_bin = cv.threshold(masked_val_img, 70, 255, cv.THRESH_BINARY)
            _, masked_sat_img_bin = cv.threshold(masked_sat_img, 40, 255, cv.THRESH_BINARY)

            # combine high val and high sat
            masked_val_sat_bin = np.zeros(sat.shape, dtype=np.uint8)
            masked_val_sat_bin[y_min:y_max, x_min:x_max] = cv.bitwise_and(masked_val_img_bin[y_min:y_max, x_min:x_max], masked_sat_img_bin[y_min:y_max, x_min:x_max])


            # count high val and high sat pixels
            image__sum = (masked_val_sat_bin > 60).sum()
            image_sums[square] = (square, image__sum)

            # keep track of greatest sum roi_corners
            if (image_sums[square][1] > greatest_sum):
                greatest_sum = image_sums[square][1]
                greatest_sum_roi_corners = roi_corners
                greatest_masked_val_sat_bin = masked_val_sat_bin

            # place button on square
            img = self.place_button_on_checkerboard(img, self.font, self.font_point[square], roi_corners, mask=masked_val_sat_bin)

        # sort sums
        image_sums = sorted(image_sums, key=itemgetter(1))
        # TODO unhard-code pixels of finger minimum
        is_finger_detected = image_sums[-1][1] > image_sums[-2][1] * 2 and image_sums[-1][1] > 500
        square_with_greatest_detection = image_sums[-1][0]
        print(str(is_finger_detected) + " " + str(square_with_greatest_detection) + " " + str(image_sums[-1][1]))
        if is_finger_detected:
            # only update cube_size if not '6th' location, that is for rotation
            button_pressed = square_with_greatest_detection + 1
            if button_pressed != 6:
                self.cube_size = button_pressed
            img = self.place_button_on_checkerboard(img, self.font_inv, self.font_point[square_with_greatest_detection], greatest_sum_roi_corners, mask=greatest_masked_val_sat_bin)
            return self.cube_size, img, button_pressed == 6
        else:
            return self.cube_size, img, False

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

    def place_button_on_checkerboard(self, img, font_img, font_point_square, roi_corners, mask):
        roi_corners_float = roi_corners.astype(np.float32)
        img = self.draw_cube.add_image_to_base(img,
                                               font_img,
                                               roi_corners_float,
                                               font_point_square[0],
                                               font_point_square[1][0],
                                               font_point_square[1][1],
                                               mask=mask)
        return img
