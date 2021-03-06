import cv2 as cv
import numpy as np
from operator import itemgetter
from draw_cube import DrawCube
from common_core import CommonCore


class ArInput:

    def __init__(self, board_width, resource_path, start_path):
        self.board_width = board_width
        self.cube_size = 1
        self.start_path = start_path
        self.resource_path = resource_path
        optional_path = "3d_cube_with_ar/"
        required_path = "font/white_rabbit_numbers.jpg"
        self.font = cv.imread("%s%s%s" % (start_path, optional_path, required_path))
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
        self.draw_cube = DrawCube(resource_path)
        self.common_core = CommonCore(None)

    def update_cube_size(self, new_cube_size):
        self.cube_size = new_cube_size

    def look_for_cube_size_v1(self, img, corners):
        cube_return = self.cube_size
        self.cube_size = (self.cube_size % 6) + 1
        return cube_return

    def look_for_cube_size_v2(self, img, corners2, draw_buttons=True):

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

            x_min, y_min, x_max, y_max = self.common_core.sort_points_for_extremes(roi_corners_for_mask)
            # smooth image
            sat[y_min:y_max, x_min:x_max] = cv.medianBlur(sat[y_min:y_max, x_min:x_max], 9)
            val[y_min:y_max, x_min:x_max] = cv.medianBlur(val[y_min:y_max, x_min:x_max], 9)

            # black out all area outside roi_corners
            masked_val_img = np.zeros(sat.shape, dtype=np.uint8)
            masked_sat_img = np.zeros(sat.shape, dtype=np.uint8)
            masked_val_img[y_min:y_max, x_min:x_max] = cv.bitwise_and(val[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max])
            masked_sat_img[y_min:y_max, x_min:x_max] = cv.bitwise_and(sat[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max])

            # threshold for high val and high sat
            _, masked_val_img[y_min:y_max, x_min:x_max] = cv.threshold(masked_val_img[y_min:y_max, x_min:x_max], 70, 255, cv.THRESH_BINARY)
            _, masked_sat_img[y_min:y_max, x_min:x_max] = cv.threshold(masked_sat_img[y_min:y_max, x_min:x_max], 40, 255, cv.THRESH_BINARY)

            # combine high val and high sat
            masked_val_sat_bin = masked_val_img
            masked_val_sat_bin[y_min:y_max, x_min:x_max] = cv.bitwise_and(masked_val_img[y_min:y_max, x_min:x_max], masked_sat_img[y_min:y_max, x_min:x_max])


            # count high val and high sat pixels
            image__sum = (masked_val_sat_bin > 60).sum()
            image_sums[square] = (square, image__sum)

            # keep track of greatest sum roi_corners
            if (image_sums[square][1] > greatest_sum):
                greatest_sum = image_sums[square][1]
                greatest_sum_roi_corners = roi_corners
                greatest_masked_val_sat_bin = masked_val_sat_bin

            # place button on square
            if draw_buttons:
                img = self.place_button_on_checkerboard(img, self.font, self.font_point[square], roi_corners, mask=masked_val_sat_bin)

        # sort sums
        image_sums = sorted(image_sums, key=itemgetter(1))
        # TODO unhard-code pixels of finger minimum
        is_finger_detected = image_sums[-1][1] > image_sums[-2][1] * 2 and image_sums[-1][1] > 500
        square_with_greatest_detection = image_sums[-1][0]
        if is_finger_detected:
            button_pressed = square_with_greatest_detection + 1
            return img, button_pressed, greatest_sum_roi_corners, greatest_masked_val_sat_bin
        else:
            return img, -1, None, None

    def place_inverted_button_on_checkerboard(self,
                                              img,
                                              font_square,
                                              greatest_masked_val_sat_bin,
                                              greatest_sum_roi_corners):
        img = self.place_button_on_checkerboard(img,
                                                self.font_inv,
                                                self.font_point[font_square],
                                                greatest_sum_roi_corners,
                                                mask=greatest_masked_val_sat_bin)
        return img

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

    def get_cube_size(self):
        return self.cube_size
