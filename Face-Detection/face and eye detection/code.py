#import all the necessary libraries
import cv2
#Add the path of .xml for face recognition
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#Add the path of .xml file for eye recognition
eye_cascade=cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
#function to capture our live video using webcam
v=cv2.VideoCapture(0)
while True:
    check,frame=v.read()
    g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(g,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=g[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('IMAGE',frame)
    if cv2.waitKey()==ord('q'):
        break
v.release()
cv2.destroyAllWindows()