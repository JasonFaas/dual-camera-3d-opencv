import numpy as np

class CommonCore:


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
