import cv2
import dlib #dlib to detect and predict the faces
import numpy as np
from collections import deque

# call back function for trackbar, creating trackbars
def nothing(x):
    print(x)
cv2.namedWindow('Lens Color')
cv2.createTrackbar('Blue','Lens Color',0,255,nothing)
cv2.createTrackbar('Green','Lens Color',0,255,nothing)
cv2.createTrackbar('Red','Lens Color',0,255,nothing)


# load the image and converting it to gray scale
img = cv2.imread('image1.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# declare detector 
detector = dlib.get_frontal_face_detector()

# detecting faces in the image
faces = detector(gray_img)
    
# locating the facial landmarks
predictor = dlib.shape_predictor('shape_predictor_70_face_landmarks.dat')

# a list to hold centre of eye coordinates
landmarksPoints = deque()

# going through all detected faces
for face in faces:
    # coordinates of a rectangle
    # x1,y1,x2,y2 = face.left(),face.top(),face.right(),face.bottom()              
    landmarks = predictor(gray_img, face)
    for n in range(68,70):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        #cv2.circle(img,(x,y),2,(0,255,0),cv2.FILLED)
        landmarksPoints.appendleft([x,y])  
               
    
    # mask prepare
    mask = np.zeros_like(img)
    print(landmarksPoints) 

    # drawing circles at centre of the eyes,(you may have to change the radius sometimes)
    leftEye = cv2.circle(mask,(landmarksPoints[0][0],landmarksPoints[0][1]),9,(255,255,255),cv2.FILLED)
    rightEye = cv2.circle(mask,(landmarksPoints[1][0],landmarksPoints[1][1]),9,(255,255,255),cv2.FILLED)

    # declaring the eye lens color(can be varied by trackbar)
    eyeColor = np.zeros_like(img,np.uint8)
    while True:
        b = cv2.getTrackbarPos('Blue','Lens Color')
        g = cv2.getTrackbarPos('Green','Lens Color')
        r = cv2.getTrackbarPos('Red','Lens Color')
        eyeColor[:] = b,g,r
        #cv2.imshow('color',eyeColor)
        eyeColorMask = cv2.bitwise_and(mask,eyeColor)
        #cv2.imshow('mask',eyeColorMask)
    
        # since the edges are pretty sharp, bluring.
        eyeColorMask = cv2.GaussianBlur(eyeColorMask,(7,7),15)

        #merging the original image and colorMask
        lens_img = cv2.addWeighted(img,1,eyeColorMask,0.4,0)
        cv2.imshow('Lens Color', lens_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
cv2.destroyAllWindows()