import cv2
import mediapipe as mp
import numpy as np
from cv2 import stylization

#Start video capture from webcam
cap = cv2.VideoCapture(0)

def cartoonify(frame):
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Applying median blur to smoothen an image
    smoothGrayScale = cv2.medianBlur(grayScale, 5)
    
    #Getting edges for cartoon effect
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
                                    cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 9)
    
    #Applying bilateral to get rid of noise
    colorImage = cv2.bilateralFilter(frame, 10, 75, 75)
    
    #Masking with edges
    finalImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    
    styleImage = cv2.stylization(finalImage, sigma_s=150, sigma_r=0.3)
    
    finalImage = cv2.addWeighted(frame, 0.4, styleImage, 0.6, 0)
    
    return finalImage

while True:
    ret, frame = cap.read()
    
    #Display final result
    cv2.imshow("Cartoon effect", cartoonify(frame))
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the video capture and close all windows
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()