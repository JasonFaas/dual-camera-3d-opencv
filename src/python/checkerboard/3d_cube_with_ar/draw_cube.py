import cv2 as cv
import numpy as np


class DrawCube:

    def draw_cube_pieces(self, corners, gray, img, board_size, cube_size):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        _, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
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
        # Read image to project and put it on image with same size as image to project to
        resources_path = '../../resources/'
        develjpg = cv.imread(resources_path + "to_project/devel_square.jpg")
        devel_jpg_full_size = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        dev_rows, dev_cols, _ = develjpg.shape
        devel_jpg_full_size[0:dev_rows, 0:dev_cols] = develjpg.copy()
        img = self.add_image_to_base(img, devel_jpg_full_size,
                                          np.array([imgpts3[0], imgpts3[3], imgpts3[2], imgpts3[1]]), dev_rows,
                                          dev_cols)
        img = self.draw_cube(img, imgpts3)
        return img

    def add_image_to_base(self, img, img_to_use, dst_pts, dev_rows, dev_cols):
        # Create matrix to transform then warp
        src_pts = np.array([
            [0, 0],
            [dev_cols - 1, 0],
            [dev_cols - 1, dev_rows - 1],
            [0, dev_rows - 1]], dtype="float32")

        matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv.warpPerspective(img_to_use, matrix, (img.shape[1], img.shape[0]))
        # create mask to project onto real image
        dst2gray = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
        _, binary_thresh = cv.threshold(dst2gray, 1, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv.morphologyEx(binary_thresh, cv.MORPH_CLOSE, kernel, iterations=3)
        final_mask_inv = cv.bitwise_not(final_mask)
        # add images together after masks applied
        background_img = cv.bitwise_and(img, img, mask=final_mask_inv)
        fg_img = cv.bitwise_and(warped_img, warped_img, mask=final_mask)
        img = cv.add(background_img, fg_img)
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