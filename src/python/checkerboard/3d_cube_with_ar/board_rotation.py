import numpy as np

class BoardRotation:

    def __init__(self, original_rotation):
        self.rotation = original_rotation

    def update_rotation(self):
        self.rotation = (self.rotation + 1) % 4

    def get_rotated_corners(self, points):
        # TODO: do default check to verify that points are in the right place

        new_points = points
        if self.rotation == 1:
            new_points = np.empty((49, 1, 2))
            for i in range(7):
                for k in range(7):
                    new_points[i + (6 - k) * 7] = points[i * 7 + k]

        if self.rotation == 2:
            new_points = np.empty((49, 1, 2))
            for i in range(7):
                for k in range(7):
                    new_points[(6 - i) * 7 + 6 - k] = points[i * 7 + k]

        if self.rotation == 3:
            new_points = np.empty((49, 1, 2))
            for i in range(7):
                for k in range(7):
                    new_points[6 + (k * 7) - i] = points[i * 7 + k]

        return new_points
