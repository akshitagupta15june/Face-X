import cv2
import dlib 
import numpy as np
from math import hypot
cap = cv2.VideoCapture(0)
mask = cv2.imread("mask/mask.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

while True:
    _,frame = cap.read()
    # frame = cv2.imread("test1.jpg")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    # print(faces)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,255),3)
        landmarks = predictor(gray, face)
        l = (landmarks.part(2).x, landmarks.part(2).y)
        r = (landmarks.part(15).x, landmarks.part(15).y)
        m = (landmarks.part(51).x, landmarks.part(51).y)
        face_width = int(hypot(l[0]-r[0],l[1]-r[1]))
        face_height = int(face_width*0.9)

        top_left= (int(m[0] - face_width/2), int(m[1]- face_height/2))
        bottom_right = (int(m[0]+face_width/2),int(m[1]+face_height/2))
        # cv2.rectangle(frame, (int(m[0] - face_width/2), int(m[1]- face_height/2)),(int(m[0]+face_width/2),int(m[1]+face_height/2)),(0,255,0),2)
        # cv2.line(frame, l,m, (0,255,0),3)
        # cv2.line(frame, m,r, (0,255,0),3)
        
        face_mask = cv2.resize(mask, (face_width, face_height))
        face_area = frame[top_left[1]: top_left[1]+ face_height,top_left[0]:top_left[0]+face_width]
        mask_gray=cv2.cvtColor(face_mask,cv2.COLOR_BGR2GRAY)
        _,face_mask2 = cv2.threshold(mask_gray, 25,255,cv2.THRESH_BINARY_INV)

        face_area_no_face = cv2.bitwise_and(face_area,face_area, mask = face_mask2)
        final_mask = cv2.add(face_area_no_face, face_mask)

        frame[top_left[1]: top_left[1]+ face_height,top_left[0]:top_left[0]+face_width]= final_mask

    cv2.imshow("FRame",frame)
    # cv2.imshow("Mask",face_mask)

    key = cv2.waitKey(1)

    if key == 27:
        break