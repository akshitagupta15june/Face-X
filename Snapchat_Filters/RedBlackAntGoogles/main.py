import cv2
import cvzone
import os

try:
    cascade_file = 'haarcascade_frontalface_default.xml'
    if not os.path.isfile(cascade_file):
        raise Exception("Cascade Classifier file does not exist")
    cascade = cv2. CascadeClassifier(cascade_file)

    overlay_file = "sunglass.png"
    if not os.path.isfile(cascade_file):
        raise Exception("Overlay image file does not exist")
    overlay = cv2.imread(overlay_file, cv2.IMREAD_UNCHANGED)

    cap= cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Error capturing frame")
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            overlay_resize = cv2.resize(overlay, (w,h))
            frame = cvzone.overlayPNG(frame, overlay_resize, [x,y])
        cv2.imshow('snap', frame) 
        if cv2.waitKey(10)== ord('q'):
            break
        cv2.waitKey(1)
    cap.release()
    cv2.destroy

except Exception as e:
    pass
