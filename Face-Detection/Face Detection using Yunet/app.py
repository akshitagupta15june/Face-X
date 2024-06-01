from utils import get_dataset
import os
import sys
import cv2
import argparse
import time
import numpy as np



parser = argparse.ArgumentParser('-p','--path','image path for processing')


args = parser.parse_args()

if not os.path.exists('./face_detection_yunet_2023mar.onnx'):
    
    print("pretrained model not available, downloading from repository")
    suc = get_dataset()

    if suc:
        pass
    else:
        print("model could not be downloaded.")

        print(sys.exis())


detector = cv2.FaceDetectorYN.create("/content/face_detection_yunet_2023mar_int8.onnx", "", (320, 320),score_threshold = 0.8)
img = cv2.imread()
img_W = int(img.shape[1])
img_H = int(img.shape[0])
detector.setInputSize((img_W, img_H))
start_time = time.time()
detections = detector.detect(img)[1]
end_time = time.time()
elapsed_time_ms = ( end_time - start_time ) * 1000
print(f"Processing time: {elapsed_time_ms:.2f} milliseconds")



def visualize_face_detections(image_path, detections):
    image = cv2.imread(image_path)

    for detection in detections:
        x, y, width, height = map(int, detection[:4])
        right_eye = tuple(map(int, detection[4:6]))
        left_eye = tuple(map(int, detection[6:8]))
        nose_tip = tuple(map(int, detection[8:10]))
        right_mouth_corner = tuple(map(int, detection[10:12]))
        left_mouth_corner = tuple(map(int, detection[12:14]))
        face_score = detection[14]

        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        cv2.circle(image, right_eye, 3, (255, 0, 0), -1)
        cv2.circle(image, left_eye, 3, (0, 0, 255), -1)
        cv2.circle(image, nose_tip, 3, (0, 255, 0), -1)
        cv2.circle(image, right_mouth_corner, 3, (255, 0, 255), -1)
        cv2.circle(image, left_mouth_corner, 3, (0, 255, 255), -1)

        cv2.putText(image, f"fs: {face_score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(image)

image_path = args.path
visualize_face_detections(image_path, detections)
