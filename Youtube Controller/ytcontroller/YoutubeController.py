import numpy as np
import cv2
import pyautogui as pagui
import time as time

# Helper function to simulate a key press
def clicker(key):
    pagui.keyDown(key) # Press the key
    time.sleep(1) # Wait for a second
    pagui.keyUp(key) # Release the key

# Helper function to detect a wink in a region of interest (ROI)
def detectWink(frame, location, ROI, cascade):
    # Detect eyes within the region of interest
    eyes = cascade.detectMultiScale(
        ROI, 1.15, 3, 0|cv2.CASCADE_SCALE_IMAGE, (10, 20)) 
    for e in eyes:
        e[0] += location[0] # Adjust for the ROI location
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        
        # Draw a rectangle around each detected eye
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    if len(eyes)==1:
        # If only one eye is detected, simulate a key press
        clicker('space')
    return len(eyes) == 1    # Return true if only one eye is detected

# Main function to detect faces and winks in real-time video
def detect(frame, faceCascade, eyesCascade):
    # Convert the frame to grayscale for processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply contrast-limited adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(18, 18))
    gray_frame = clahe.apply(gray_frame)
    
    # Apply a bilateral filter to smooth the image while preserving edges
    gray_frame = cv2.bilateralFilter(gray_frame, 5, 15, 15)

    # Detect faces in the grayscale frame
    scaleFactor = 1.05 # Range is from 1 to ..
    minNeighbors = 2   # Range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # Either 0 (faster) or 0|cv2.CASCADE_SCALE_IMAGE (more accurate)
    minSize = (30,30) # Range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)

    detected = 0
    for f in faces:
        # Get the coordinates of the face
        x, y, w, h = f[0], f[1], f[2], f[3]
        # Extract the region of interest (ROI) for the face
        faceROI = gray_frame[y:y+h, x:x+w]
        # Detect winks within the ROI
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1
            # Draw a blue rectangle around the face if a wink is detected
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        else:
            # Draw a green rectangle around the face if no wink is detected
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected

# Main function to run the video-based wink


def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(1) == ord('q'):
            showlive = False 
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_eye.xml')
    runonVideo(face_cascade, eye_cascade)
