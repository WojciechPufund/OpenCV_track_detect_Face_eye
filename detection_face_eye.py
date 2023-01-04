'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Face and eye detection aplication
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Author:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Wojciech Pufund
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

PLease choice detection:
1. Camera
2. Video (*.mp4)

3. Close app
'''
import cv2
import sys

print(__doc__)

choice = input("Choice: ").lower()
print()

if choice == '1' or choice == 'camera':
    print('Starting the camera...')
    cascPath = "haarcascade_frontalface_default.xml"
    eyePath = "haarcascade_eye.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    eyeCascade = cv2.CascadeClassifier(eyePath)
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eyeCascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey),(ex+ew, ey+eh), (0, 0, 255), 2)
        cv2.imshow(
            'Detect face & eye from camera by Wojciech Pufund. For exit press (q)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

elif choice == '2' or choice == 'video':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture('video.mp4')
    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow(
            'Detect face & eye from video by Wojciech Pufund. For exit press (q)', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

elif choice == '3' or choice == 'close':
    print("EXIT")
    pass
