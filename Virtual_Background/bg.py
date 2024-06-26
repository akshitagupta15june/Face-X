import cv2
import mediapipe as mp
import numpy as np
import time 

print("All modules imported!")
#Set variables
threshold=0.5
b_amt=15
fps=0

#Load image segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

print("Starting video capture...")
#Start video capture from webcam
cap = cv2.VideoCapture(0)

#Find out frame size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Load and resize background
background_img = cv2.imread("bg.jpg")
background_img =  cv2.resize(background_img,(width,height))

#Apply gaussian blur to background
bg_image = cv2.GaussianBlur(background_img, (b_amt,b_amt), 0) #[:,:,::-1]

print("Capture started!")
start_time =  time.time()
frame_count=0
while True:
    #Read frame from video stream
    ret, frame= cap.read()
    
    #Create a mask for person and background
    results = selfie_segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    segmentation_mask = results.segmentation_mask

    #Process mask using threshold value
    condition = np.stack((segmentation_mask,) * 3, axis=-1)  > threshold
    
    #Use the mask to apply background to frame
    output_image = np.where(condition, frame, bg_image)
    
    #Count and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:  # Update FPS every second
        fps = frame_count / elapsed_time
        print(f"FPS: {round(fps, 2)}")
        start_time = time.time()
        frame_count = 0
    cv2.putText(output_image, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #Display final result
    cv2.imshow("Virtual background",output_image)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()