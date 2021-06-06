import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from collections import deque

def filter(frame,landmarks):
    crownImg = cv2.imread("wanda.png",-1)
    crownMask = crownImg[:, :, 3] # binary image
    crownMaskInv = cv2.bitwise_not(crownMask) # inverting the binary img
    crownImg = crownImg[:, :, 0:3]

    # dimensions of the crown
    crownHt, crownWd  = crownImg.shape[:2]

    # adjusting dimensions according to the landmarks
    crownWd1 = abs(landmarks.part(17).x - landmarks.part(26).x) + 50
    crownHt1 = int(crownWd1 * crownHt / crownWd)
    
    # resize the crown img
    crown = cv2.resize(crownImg, (crownWd1, crownHt1), cv2.INTER_AREA)
    mask = cv2.resize(crownMask, (crownWd1, crownHt1), cv2.INTER_AREA)
    mask_inv = cv2.resize(crownMaskInv, (crownWd1, crownHt1), cv2.INTER_AREA)

    # grab the region of interest and apply the crown
    x1 = int(landmarks.part(27).x - (crownWd1 / 2))
    y1 = int(landmarks.part(24).y - 110)
    x2 = int(x1 + crownWd1)
    y2 = int(y1 + crownHt1)
    
    roi = frame[y1:y2, x1:x2]
    backGround = cv2.bitwise_and(roi, roi, mask=mask_inv).astype('uint8')
    foreGround = cv2.bitwise_and(crown, crown, mask=mask).astype('uint8')
    
    # Adding filter to the frame
    frame[y1:y2, x1:x2] = cv2.add(backGround, foreGround)

    return frame


#declaring the detector
detector = dlib.get_frontal_face_detector()
# locating the facial landmarks
predictor = dlib.shape_predictor('shape_predictor_70_face_landmarks.dat')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecting faces in the frame
        faces = detector(gray)

        # going through all detected faces
        for face in faces:
            landmarksPoints = deque()
            landmarks = predictor(gray, face)
            frame = filter(frame, landmarks)

            # grabbing eye coordinates
            for n in range(68,70):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                #cv2.circle(frame,(x,y),1,(0,255,255),1)    
                landmarksPoints.appendleft([x,y]) 

            # mask prepare
            eyeMask = np.zeros_like(frame)
            
            # drawing circles at centre of the eyes
            leftEye = cv2.circle(eyeMask,(landmarksPoints[0][0],landmarksPoints[0][1]),5,(255,255,255),cv2.FILLED)
            rightEye = cv2.circle(eyeMask,(landmarksPoints[1][0],landmarksPoints[1][1]),5,(255,255,255),cv2.FILLED)
            
            # declaring the eye lens color as red
            eyeColor = np.zeros_like(frame,np.uint8)
            b,g,r = 0,0,255
            eyeColor[:] = b,g,r
            
            # merging the original image and colorMask
            eyeColorMask = cv2.bitwise_and(eyeMask,eyeColor)
            eyeColorMask = cv2.GaussianBlur(eyeColorMask,(7,7),15)
            frame = cv2.addWeighted(frame,1,eyeColorMask,0.4,0)
        
        cv2.imshow('Wanda Filter', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
