import cv2
import mediapipe as mp
from math import hypot
import itertools

cap = cv2.VideoCapture(0)

nose_img = cv2.imread('bullnose.png') # (860, 563) w,h ratio=563/860=0.65

prevTime = 0

nose_landmarks = [49,279,197,2,5] # 5 = center nose point
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4) # max_num_faces=1
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480)) #(1200, 650)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #nose landmarks
            leftnosex, leftnosey, rightnosex, rightnosey, centernosex, centernosey = 0, 0, 0, 0, 0, 0
            #landmark info
            for lm_id, lm in enumerate(face_landmarks.landmark):
                
                #original values
                h, w, c = rgb.shape
                x, y = int(lm.x * w), int(lm.y * h)
                
                #calculating nose width
                if lm_id == nose_landmarks[0]:
                    leftnosex, leftnosey = x, y
                if lm_id == nose_landmarks[1]:
                    rightnosex, rightnosey = x, y
                if lm_id == nose_landmarks[4]:
                    centernosex, centernosey = x, y

            nose_width = int(hypot(leftnosex-rightnosex, leftnosey-rightnosey*1.2))
            nose_height = int(nose_width*0.77)
            
            if (nose_width and nose_height) != 0:
                pig_nose = cv2.resize(nose_img, (nose_width, nose_height))

            top_left = (int(centernosex-nose_width/2),int(centernosey-nose_height/2))
            bottom_right = (int(centernosex+nose_width/2),int(centernosey+nose_height/2))
            
            nose_area = frame[
                top_left[1]: top_left[1]+nose_height,
                top_left[0]: top_left[0]+nose_width
            ]
            #creating nose mask
            pig_nose_gray = cv2.cvtColor(pig_nose, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(pig_nose_gray, 25, 255, cv2.THRESH_BINARY_INV)
            #removing nose
            no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            #superimposing nose on no_nose
            final_nose = cv2.add(no_nose, pig_nose)
            #putting pig nose filter on original nose
            frame[
                top_left[1]: top_left[1]+nose_height,
                top_left[0]: top_left[0]+nose_width
            ] = final_nose

    cv2.imshow("output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
