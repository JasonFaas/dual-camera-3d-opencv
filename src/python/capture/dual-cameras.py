import cv2 as cv 

cam0 = cv.VideoCapture(0)
cam1 = cv.VideoCapture(1)

cv.namedWindow("Zero Test")
cv.namedWindow("One Test")

img_counter = 0

print("Hotkeys:")
quit = 'q'
screenshot = 's'
start_recording = 'r'
stop_recording = 't'
print("\t" + 'quit:' + quit)
print("\t" + 'screenshot:' + screenshot)
print("\t" + 'start_recording:' + start_recording)
print("\t" + 'stop_recording:' + stop_recording)

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()
    cv.imshow("Zero Test", frame0)
    cv.imshow("One Test", frame1)
    if not ret0 or not ret1:
        break
        
    # Display the resulting frame
    key_press = cv.waitKey(1) & 0xFF
    if key_press == ord(quit):
        print("Escape hit, closing...")
        break
    elif key_press == ord(screenshot):
        # SPACE pressed
        img_name_0 = "opencv_frame_{}_0.png".format(img_counter)
        img_name_1 = "opencv_frame_{}_1.png".format(img_counter)
        cv.imwrite(img_name_0, frame0)
        cv.imwrite(img_name_1, frame1)
        print("{} written!".format(img_name_0))
        print("{} written!".format(img_name_1))
        img_counter += 1

cam0.release()
cam1.release()

cv.destroyAllWindows()
