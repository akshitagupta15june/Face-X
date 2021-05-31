import numpy as np
import cv2
import sqlite3

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def InsertOrUpdate(Id,Name):
    conn=sqlite3.connect("facebase.db")
    cmd="SELECT * FROM People WHERE Id="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE people SET Name=' "+str(name)+" ' WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO people(ID,Name) Values("+str(Id)+",' "+str(name)+" ' )"


    conn.execute(cmd)
    conn.commit()
    conn.close()

id = input('Enter User_id : ')
name=input('Enter your Name : ')
InsertOrUpdate(id,name)
num=0

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        num=num+1
        color = (0,255,0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.imwrite("dataSet/User."+str(id)+"."+str(num)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)
        cv2.waitKey(100)
    cv2.imshow('Face',frame)
    cv2.waitKey(1)
    if(num>20):
        break
cap.release()
cv2.destroyAllWindows()
