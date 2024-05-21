# Hand Navigation Landmark

Full Name : Utsab Samadder
Email : utsab.samadder@gmail.com

**How to Run it**
1. Run the FaceDetectionYolov8.ipynb file on Google Colab
2. Once all the block are run then download data.yaml and best.pt file
3. Use those to file to run the model with FaceDetection.py to do it in real time


**The code for FaceDetection.py**
```python
from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture(0)  # For Video
#cap = cv2.VideoCapture(r"C:\Users\USER\Downloads\Video\Pothole.mp4")  # For Video
model = YOLO("best.pt")

#from data.yaml file copy the classes
classNames = ['Face']
myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf>0.50:
                if currentClass =='Face' :
                    myColor = (0, 255, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=2,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=3)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
```

**Couldn't upload the repo here because size is more than 2 gb**

[Dataset Link](https://universe.roboflow.com/yolo-ccdgk/face-detect-i4e0c/dataset/4)