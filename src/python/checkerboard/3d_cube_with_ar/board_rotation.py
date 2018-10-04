import numpy as np
from common_core import CommonCore

class BoardRotation:

    def __init__(self, original_rotation):
        self.last_0th_checkerboard_point = [-1, -1]
        self.common_core = CommonCore()

    def update_rotation(self, last_point):
        self.last_0th_checkerboard_point = last_point

    def get_rotated_corners(self, points):
        # TODO: do default check to verify that points are in the right place

        rotation = 0
        new_points = points
        x_min, y_min, x_max, y_max = self.common_core.sort_points_for_extremes(new_points)
        distance_between_checkerboard_extremes = self.distance_between_points([x_min, y_min], [x_max, y_max])
        if self.relative_0th_change(distance_between_checkerboard_extremes, new_points[0, 0], self.last_0th_checkerboard_point) > .2:
            if self.relative_0th_change(distance_between_checkerboard_extremes, new_points[6, 0], self.last_0th_checkerboard_point) < .2:
                rotation = 1
            elif self.relative_0th_change(distance_between_checkerboard_extremes, new_points[7 * 6, 0], self.last_0th_checkerboard_point) < .2:
                rotation = 3
            elif self.relative_0th_change(distance_between_checkerboard_extremes, new_points[7 * 6 + 6, 0], self.last_0th_checkerboard_point) < .2:
                rotation = 2

        # TODO make this more efficient
        if rotation == 1:
            new_points = np.empty((49, 1, 2))
            for i in range(7):
                for k in range(7):
                    new_points[i + (6 - k) * 7] = points[i * 7 + k]

        if rotation == 2:
            new_points = np.empty((49, 1, 2))
            for i in range(7):
                for k in range(7):
                    new_points[(6 - i) * 7 + 6 - k] = points[i * 7 + k]

        if rotation == 3:
            new_points = np.empty((49, 1, 2))
            for i in range(7):
                for k in range(7):
                    new_points[6 + (k * 7) - i] = points[i * 7 + k]



        self.last_0th_checkerboard_point = new_points[0, 0]

        return new_points

    def relative_0th_change(self, distance_between_checkerboard_extremes, this_0th, last_0th):
        distance_between_start_points = self.distance_between_points(last_0th, this_0th)
        relative_0th_change = distance_between_start_points / distance_between_checkerboard_extremes
        return relative_0th_change

    def distance_between_points(self, point_1, point_2):

        distance_between_start_points = ((point_1[0] - point_2[0]) ** 2 + (
                point_1[1] - point_2[1]) ** 2) ** 0.5
        return distance_between_start_points
