import sys
import dlib
import cv2

# Load the cnn face detector
detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3

while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    # Perform inference on frame
    dets = detector(rgb_image,1)

    # Plot the bounding boxes on frame
    for det in dets:
        cv2.rectangle(img,(det.rect.left(), det.rect.top()), (det.rect.right(), det.rect.bottom()), color_green, line_width)
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()

'''
Run: python3 cnn_face_detect.py
Note: Requires GPU for real-time inference
'''
