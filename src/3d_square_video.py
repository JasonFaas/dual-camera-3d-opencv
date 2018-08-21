import numpy as np
import cv2 as cv
import glob
import datetime

def add_square_to_img(img_original):

    board_size = (7, 7)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_size[1]*board_size[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Read image
    img = img_original.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, board_size, None)

    # If not found return
    if not ret:
        print('no checkerboard')
        return img_original

    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners)

    # Draw and display the corners
    cv.drawChessboardCorners(img, board_size, corners, ret)



    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = img_original.copy()
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('calibresult.png',dst)


    #3d Time
    criteria2 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp2 = np.zeros((board_size[1]*board_size[0],3), np.float32)
    objp2[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
    axis2 = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    # Find the rotation and translation vectors.
    ret2,rvecs2, tvecs2 = cv.solvePnP(objp2, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts2, jac2 = cv.projectPoints(axis2, rvecs2, tvecs2, mtx, dist)

    print('f')
    print(corners2)
    print('g')
    print(imgpts2)
    def draw2(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img
    # img = draw2(img,corners2,imgpts2)




    axis3 = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                       [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
    ret2,rvecs2, tvecs2 = cv.solvePnP(objp2, corners2, mtx, dist)
    imgpts3, jac3 = cv.projectPoints(axis3, rvecs2, tvecs2, mtx, dist)

    def draw3(img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        # draw ground floor in green
        img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
        # draw pillars in blue color
        for i,j in zip(range(4),range(4,8)):
            img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        # draw top layer in red color
        img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
        return img

    img = draw3(img,corners2,imgpts3)

    return img


camera = 0
cap = cv.VideoCapture(camera)
camera_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

recording = False

# out.write(frame)

while(True):
    # Capture frame-by-frame
    retval, frame = cap.read()
    if retval:
        # Our operations on the frame come here
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        frame = add_square_to_img(frame)

        # Display the resulting frame
        key_press = cv.waitKey(1) & 0xFF
        if key_press == ord('q'):
            if recording:
                out.release()
            break
        elif key_press == ord('s'):
            datetime_now = str(datetime.datetime.now()).replace(' ', '_')
            img_name = "opencv_frame_{}_{}.png".format(datetime_now, str(camera))
            cv.imwrite(img_name, frame)

        elif key_press == ord('r') and not recording:
            recording = True
            datetime_now = str(datetime.datetime.now()).replace(' ', '_')
            video_name = "opencv_video_{}_{}.mkv".format(datetime_now, str(camera))
            fourcc = cv.VideoWriter_fourcc(*'X264')
            out = cv.VideoWriter(video_name, fourcc, 20, (camera_width, camera_height))
        elif key_press == ord('t') and recording:
            recording = False
            out.release()

        if recording:
            out.write(frame)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, 'Recording', (10, 500), font, 4, (100, 100, 255), 4, cv.LINE_AA)

        cv.imshow('frame',frame)

cap.release()
cv.destroyAllWindows()
