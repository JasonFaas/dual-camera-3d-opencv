import cv2 as cv
import numpy as np

class DrawCube:


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