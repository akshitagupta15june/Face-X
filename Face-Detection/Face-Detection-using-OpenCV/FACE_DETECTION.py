#FACE DETECTION CV
import cv2
CLASSIFIER_PATH = "Facedetection/haarcascade_frontalface_default.xml"  #Path of the file 
cam = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(CLASSIFIER_PATH)
while(True):
    _, frame = cam.read()
    #Converting to Grayscale img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Returns coordinates of all faces in the frame 
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    #cyclye through each coordinate list 
    for face_dims in faces:
        #Desctructing data and extracted bounding box coordinates
        (x,y,w,h) = face_dims
        mid_x = int(x + h/2)
        mid_y = int(y+ h/2)
        #Drawing -"Bounding Box"
        frame = cv2.rectangle(frame, (x,y), (x+h, y+h), (0,255,255), 2)
        frame = cv2.putText(frame, str(x), (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.7,(0,0,255), 2)  
        frame = cv2.putText(frame,"Mid", (mid_x, mid_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255),2)  
    #Displaying -"Bounding Box"
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if(key == 27):
        break

cam.release()
cv2.destroyAllWindows()   
