import os
import cv2

face_cascade_path='../data/cascade/haarcascade_frontalface_default.xml'
images_path='../data/images/'

def detect_faces():
    face_cascade=cv2.CascadeClassifier(face_cascade_path)
    os.chdir(images_path)
    
    if len(os.listdir())==0:
        print('\n\nNo Images Found')
        input()
        quit()
    for i in os.listdir():
        name=i
        img=cv2.imread(i)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
    
        for (x,y,w,h) in faces:
            image=img[y:y+h,x:x+w]
            output_name="{}.faces{}x{}_{}x{}.jpg".format(name,x,y,w,h)
            cv2.imwrite('../faces/'+output_name,image)
            print('Done Detecting: ',output_name)
    os.chdir('../../script')