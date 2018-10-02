import cv2 as cv
import datetime

camera = 0
cap = cv.VideoCapture(camera)
camera_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

recording = False

# out.write(frame)

print("Hotkeys:")
quit = 'q'
screenshot = 's'
start_recording = 'r'
stop_recording = 't'
print("\t" + 'quit:' + quit)
print("\t" + 'screenshot:' + screenshot)
print("\t" + 'start_recording:' + start_recording)
print("\t" + 'stop_recording:' + stop_recording)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Our operations on the frame come here
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Display the resulting frame
        key_press = cv.waitKey(1) & 0xFF
        if key_press == ord(quit):
            if recording:
                out.release()
            break
        elif key_press == ord(screenshot):
            datetime_now = str(datetime.datetime.now()).replace(' ', '_')
            img_name = "opencv_frame_{}_{}.png".format(datetime_now, str(camera))
            cv.imwrite(img_name, frame)

        elif key_press == ord(start_recording) and not recording:
            recording = True
            datetime_now = str(datetime.datetime.now()).replace(' ', '_')
            video_name = "opencv_video_{}_{}.mkv".format(datetime_now, str(camera))
            fourcc = cv.VideoWriter_fourcc(*'X264')
            out = cv.VideoWriter(video_name, fourcc, 20, (camera_width, camera_height))
        elif key_press == ord(stop_recording) and recording:
            recording = False
            out.release()

        if recording:
            out.write(frame)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, 'Recording', (10, 500), font, 4, (100, 100, 255), 4, cv.LINE_AA)

        cv.imshow('frame',frame)

cap.release()
cv.destroyAllWindows()
