import cv2 as cv
import numpy as np


class ArInput:

    def __init__(self):
        self.cube_size = 1

    def look_for_cube_size(self, img, corners):
        cube_return = self.cube_size
        self.cube_size = (self.cube_size % 6) + 1
        return cube_return