import cv2

cam0 = cv2.VideoCapture(2)
cam1 = cv2.VideoCapture(0)

cv2.namedWindow("Zero Test")
cv2.namedWindow("One Test")

img_counter = 0

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()
    cv2.imshow("Zero Test", frame0)
    cv2.imshow("One Test", frame1)
    if not ret0 or not ret1:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name_0 = "opencv_frame_{}_0.png".format(img_counter)
        img_name_1 = "opencv_frame_{}_1.png".format(img_counter)
        cv2.imwrite(img_name_0, frame0)
        cv2.imwrite(img_name_1, frame1)
        print("{} written!".format(img_name_0))
        print("{} written!".format(img_name_1))
        img_counter += 1

cam0.release()
cam1.release()

cv2.destroyAllWindows()
