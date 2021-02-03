import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
data_path='/home/akshita/Desktop/Face_reco/'
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_data,Labels=[],[]

for i, files in enumerate(onlyfiles):
    image_path=data_path + onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)
Labels=np.asarray(Labels,dtype=np.int32)

#model=cv2.face.EigenFaceRecognizer_create()
model=cv2.face.FisherFaceRecognizer_create()
model.train(np.asarray(Training_data),np.asarray(Labels))
print("Model Training Complete")


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is():
        return img,[]
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
            display_string = str(confidence) +'% Confidence it is user'
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

        if confidence > 75:
            cv2.putText(image, "UNLOCKED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
        else:
            cv2.putText(image, "LOCKED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()


