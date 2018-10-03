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
        font_one_points = [[16, 155], [80, 80]]
        font_two_points = [[20, 155+80+50], [80, 80]]
        font_three_points = [[20, 155+(80+50) * 2], [80, 80]]
        font_four_points = [[22, 155+(80+50) * 3 - 5], [80, 80]]
        font_five_points = [[22, 155+(80+50) * 4 - 10], [80, 80]]
        font_six_points = [[18, 155+(80+50) * 5 - 5], [80, 80]]
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
        smallest_x = np.min(corners2[:, 0, 0])
        smallest_y = np.min(corners2[:, 0, 1])
        largest_x = np.max(corners2[:, 0, 0])
        largest_y = np.max(corners2[:, 0, 1])
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
        for square in range(6):
            first_bad_format = corners2[0 + self.board_width * square, 0]
            second_bad_format = corners2[1 + self.board_width * square, 0]
            fourth_bad_format = corners2[self.board_width + self.board_width * square, 0]
            third_bad_format = corners2[self.board_width + 1 + self.board_width * square, 0]
            first = (first_bad_format[0], first_bad_format[1])
            second = (int(first[0] * 2 - second_bad_format[0]), int(first[1] * 2 - second_bad_format[1]))
            fourth = (fourth_bad_format[0], fourth_bad_format[1])
            third = (int(fourth[0] * 2 - third_bad_format[0]), int(fourth[1] * 2 - third_bad_format[1]))
            roi_corners = np.array([[first, second, third, fourth]], dtype=np.int32)



            # draw lines
            # cv.line(img, first, second, (255, 0, 0), thickness=5)
            # cv.line(img, second, third, (255, 255, 0), thickness=5)
            # cv.line(img, third, fourth, (255, 0, 255), thickness=5)
            # cv.line(img, fourth, first, (0, 0, 0), thickness=5)

            # TODO: review this, it may not be working like I think (though it is working)
            mask = np.zeros(sat.shape, dtype=np.uint8)
            # channel_count = sat.shape[2]
            channel_count = 1
            ignore_mask_color = (255,) * channel_count
            cv.fillPoly(mask, roi_corners, ignore_mask_color)
            masked_val_img = cv.bitwise_and(val, mask)
            _, masked_val_img_bin = cv.threshold(masked_val_img, 100, 255, cv.THRESH_BINARY)
            masked_sat_img = cv.bitwise_and(sat, mask)
            masked_sat_img = cv.bitwise_and(sat, masked_val_img_bin)
            # cv.imshow('just_focus_' + str(square), masked_sat_img)
            image__sum = (masked_sat_img > 60).sum()
            image_sums[square] = (square, image__sum)
            # print(str(square) + "_" + str(image__sum))

            # TODO draw ar button on checkerboard
            # place button on square
            roi_corners = np.array([[third], [second], [first], [fourth]], dtype=np.int32)
            roi_corners_float = roi_corners.astype(np.float32)
            img = self.draw_cube.add_image_to_base(img, self.font, roi_corners_float, self.font_point[square][0], self.font_point[square][1][0], self.font_point[square][1][1])

        image_sums = sorted(image_sums, key=itemgetter(1))
        is_finger_detected = image_sums[-1][1] > image_sums[-2][1] * 2
        print(str(is_finger_detected) + " " + str(image_sums[-1][0]))
        if is_finger_detected:
            # highlight_finger(image_sums[-1][0])
            self.cube_size = image_sums[-1][0] + 1
            return self.cube_size, img
        else:
            return self.cube_size, img
