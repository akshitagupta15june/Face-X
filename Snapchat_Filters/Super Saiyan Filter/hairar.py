import cv2
import dlib
import numpy as np
from math import hypot

# Load custom hair image with alpha mask
hair_image = cv2.imread("ssj_hair.png")
hair_mask = cv2.imread("ssj_hair.png", cv2.IMREAD_UNCHANGED)[...,3]

# Load golden aura sprite images (animation)
glow_imgs = np.load('glow_mini.npy')

# Load black hair image with alpha mask
black_hair = cv2.imread("black_hair.png")
black_hair_mask = cv2.imread("black_hair.png", cv2.IMREAD_UNCHANGED)[...,3]

# Initialize video capturer
cap = cv2.VideoCapture(0)

# Loading face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Add hair overlay image
def merge(frame, hair, mask, centre):

  frows, fcols, _ = frame.shape
  hrows, hcols, _ = hair.shape
  
  x, y = (centre[0] - hcols//2, centre[1] -  hrows//2)
 
  frame_roi = frame[ y:y+hrows, x:x+hcols ,... ]

  # Check bounds and overalay hair
  if frame_roi.shape == hair.shape:  
    np.copyto(frame_roi, hair,where=mask[...,np.newaxis]>0)

# Add glow overlay image
def add_glow(frame, glow, origin):
    
   frows, fcols, _ = frame.shape
   grows, gcols, _ = glow.shape
   
   x, y = origin

   frame_roi = frame[ y:y+grows, x:x+gcols ,... ]
   
   # Check bounds and overlay aura glow
   if frame_roi.shape ==  glow.shape:
    glow_roi = cv2.addWeighted(frame_roi,1.0, glow,0.95,0)
    np.copyto(frame_roi, glow_roi)

# Check if mouth is open
def mouth_open(landmarks):
    
    l1 =   landmarks.part(50).y - landmarks.part(61).y
    l2 =   landmarks.part(51).y - landmarks.part(62).y
    l3 =  landmarks.part(52).y - landmarks.part(63).y
    
    m1 =  landmarks.part(61).y - landmarks.part(67).y
    m2 =  landmarks.part(62).y - landmarks.part(66).y
    m3 =  landmarks.part(63).y - landmarks.part(65).y

    # Calculate average mouth and lip heights
    lip_height = abs((l1+l2+l3)//3)
    mouth_height = abs((m1+m2+m3)//3)
    
    if (mouth_height > lip_height*1.2):
       # mouth is open
       return True
    else:
       # mouth is closed
       return False

# Initialize frame counter
frame_count = 1

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_count = frame_count + 1

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces from the grayscale image
    faces = detector(gray)
   
    if len(faces) == 1:

        # Predict facial landmark points on the frame
        landmarks = predictor(gray, faces[0])

        # Find the extreme landmark points
        left_face = (landmarks.part(0).x, landmarks.part(0).y)
        right_face = (landmarks.part(16).x, landmarks.part(16).y)


        # Choose yellow hair if mouth is open
        if (mouth_open(landmarks)):
           hair_im = hair_image
           hair_mk = hair_mask
           hwidth_multiplier = 2
           htop_multiplier = 2
           ssj = True

        else:
           hair_im = black_hair
           hair_mk = black_hair_mask
           hwidth_multiplier = 2.6
           htop_multiplier = 3
           ssj = False           

        # Configure hair width and height
        hair_width = int(hypot(left_face[0] - right_face[0],
                           left_face[1] - right_face[1])*hwidth_multiplier)
        hair_height = int(hair_width * 0.9)

        # Resize hair and mask
        hair =  cv2.resize(hair_im, (hair_width, hair_height))
        mask =  cv2.resize(hair_mk, (hair_width, hair_height), interpolation = cv2.INTER_NEAREST)

        # Compute hair location
        hair_x = landmarks.part(27).x
        hair_y = int(landmarks.part(27).y  + (landmarks.part(27).y - landmarks.part(30).y )*htop_multiplier)

        merge(frame, hair, mask, (hair_x, hair_y))

        # Resize the aura image frame
        glow =  cv2.resize(glow_imgs[frame_count%4], (int(hair_width*2.4), int(hair_width*3.4) ))

        # Compute the hair location
        glow_x_left = landmarks.part(27).x - int(hair_width * 1.2)
        glow_y_top = int(landmarks.part(27).y  + (landmarks.part(27).y - landmarks.part(30).y )*7)
         
        # Crop the aura image based on height 
        glow = glow[0:frame.shape[0]-glow_y_top, ...]
        if ssj:
          add_glow(frame, glow, (glow_x_left , glow_y_top))
         
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
