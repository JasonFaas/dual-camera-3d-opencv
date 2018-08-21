import numpy as np
import cv2
import glob


def undistort_this(good_fname):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # print("mtx: " + str(mtx))
    # print("dist: " + str(dist))
    np.savez("0_distorition_info.txt", mtx)
    print(mtx)

    img = cv2.imread(good_fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # print("ncm: " + str(newcameramtx))
    # print("roi: " + str(roi))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('calibresult.png',dst)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

board_size = (7, 7)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_size[1]*board_size[0],3), np.float32)
objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images_0 = glob.glob('old_picture_hold/1/opencv*.png')
images_1 = glob.glob('*_1.png')

good_fname = ''

for fname in images_0:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow('img',img)
        print("pass_" + fname)
        while True:
            k = cv2.waitKey(10)
            if k % 256 == 27:
                break

        undistort_this(fname)
    else:
        print("fail")





cv2.destroyAllWindows()