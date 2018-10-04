import cv2 as cv
import numpy as np
import math
from common_core import CommonCore

class DrawCube:

    def __init__(self):
        resources_path = '../../resources/'
        self.bottom_cube_img = cv.imread(resources_path + "to_project/devel_square.jpg")
        self.common_core = CommonCore()

    def draw_cube_pieces(self, corners, corners2, img, board_size, cube_size):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # Arrays to store object points and image points from all the images.
        objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        objpoints.append(objp)
        imgpoints.append(corners)

        _, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, (img.shape[1],img.shape[0]), None, None)
        # 3d Time
        objp2 = np.zeros((board_size[1] * board_size[0], 3), np.float32)
        objp2[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        edge_size = cube_size
        x_offset = 0
        y_offset = 0
        z_offset = -0
        axis3 = np.float32([[x_offset, y_offset, z_offset], [x_offset, y_offset + edge_size, z_offset],
                            [x_offset + edge_size, y_offset + edge_size, z_offset],
                            [x_offset + edge_size, y_offset, z_offset],
                            [x_offset, y_offset, z_offset - edge_size],
                            [x_offset, y_offset + edge_size, z_offset - edge_size],
                            [x_offset + edge_size, y_offset + edge_size, z_offset - edge_size],
                            [x_offset + edge_size, y_offset, z_offset - edge_size]])
        _, rvecs2, tvecs2 = cv.solvePnP(objp2, corners2, mtx, dist)
        imgpts3, jac3 = cv.projectPoints(axis3, rvecs2, tvecs2, mtx, dist)

        destination_points = np.array([imgpts3[0], imgpts3[3], imgpts3[2], imgpts3[1]])


        # Read image to project and put it on image with same size as image to project to
        project_img = self.bottom_cube_img
        dev_rows, dev_cols, _ = project_img.shape

        img = self.add_image_to_base(img, project_img, destination_points, [0,0], dev_cols, dev_rows)
        img = self.draw_cube(img, imgpts3)
        return img

    def add_image_to_base(self, img, project_img, dst_pts, src_start_point, src_width, src_height, mask=None):

        # Put image passed in onto black image
        img_to_use = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        mask_to_use = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        img_to_use[0:src_width, 0:src_height] = \
            project_img[src_start_point[0]:src_start_point[0] + src_width, src_start_point[1]:src_start_point[1] + src_height]

        # Create matrix to transform then warp
        src_pts = np.array([
            [0,0],
            [src_width, 0],
            [src_width, src_height],
            [0, src_height]], dtype="float32")

        matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv.warpPerspective(img_to_use, matrix, (img.shape[1], img.shape[0]))

        dst_pts_int = dst_pts.astype(int)
        dst_pts_int[2][0][0] = math.ceil(dst_pts[2][0][0])
        dst_pts_int[2][0][1] = math.ceil(dst_pts[2][0][1])
        dst_pts_int[3][0][0] = math.ceil(dst_pts[3][0][0])
        dst_pts_int[3][0][1] = math.ceil(dst_pts[3][0][1])

        # Points for roi
        x_min, y_min, x_max, y_max = self.common_core.sort_points_for_extremes(dst_pts_int)

        # create mask with white quadrangle around destination points
        cv.fillConvexPoly(mask_to_use, dst_pts_int, 255)
        mask_to_use_inv = cv.bitwise_not(mask_to_use)

        if type(mask) != type(None):
            cv.imshow('mask_to_use_inv', mask_to_use_inv[y_min:y_max, x_min:x_max])
            mask_to_use_inv[y_min:y_max, x_min:x_max] = cv.bitwise_or(mask[y_min:y_max, x_min:x_max], mask_to_use_inv[y_min:y_max, x_min:x_max])

        img[y_min:y_max, x_min:x_max] = cv.bitwise_and(img[y_min:y_max, x_min:x_max], img[y_min:y_max, x_min:x_max], mask=mask_to_use_inv[y_min:y_max, x_min:x_max])
        mask_to_use_inv_inv = cv.bitwise_not(mask_to_use_inv)
        warped_img[y_min:y_max, x_min:x_max] = cv.bitwise_and(warped_img[y_min:y_max, x_min:x_max], warped_img[y_min:y_max, x_min:x_max], mask=mask_to_use_inv_inv[y_min:y_max, x_min:x_max])
        img[y_min:y_max, x_min:x_max] = cv.add(img[y_min:y_max, x_min:x_max], warped_img[y_min:y_max, x_min:x_max])

        return img

    def draw_cube(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # draw back left wall
        back_pts_2 = np.empty((0, 2), dtype=int)
        back_pts_2 = np.append(back_pts_2, [imgpts[0]], axis=0)
        back_pts_2 = np.append(back_pts_2, [imgpts[4]], axis=0)
        back_pts_2 = np.append(back_pts_2, [imgpts[5]], axis=0)
        back_pts_2 = np.append(back_pts_2, [imgpts[1]], axis=0)
        img = cv.drawContours(img, [back_pts_2], -1, (100, 100, 255), 10)

        # draw back right wall
        back_pts_2 = np.empty((0, 2), dtype=int)
        back_pts_2 = np.append(back_pts_2, [imgpts[0]], axis=0)
        back_pts_2 = np.append(back_pts_2, [imgpts[4]], axis=0)
        back_pts_2 = np.append(back_pts_2, [imgpts[7]], axis=0)
        back_pts_2 = np.append(back_pts_2, [imgpts[3]], axis=0)
        img = cv.drawContours(img, [back_pts_2], -1, (255, 100, 100), -1)

        # draw pillars in blue color
        for i, j in zip(range(0, 4), range(4, 8)):
            img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

        # draw top layer in red color
        img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        return img