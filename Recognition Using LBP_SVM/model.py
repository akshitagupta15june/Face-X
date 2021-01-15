import cv2
import os
import glob
from skimage import feature
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class LocalBinaryPatterns:

    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


desc = LocalBinaryPatterns(24, 8)
img_folder = 'dataset/'

labels = []
array = []

i = 0
cnt = 0
mapping = {}
for dir1 in os.listdir(img_folder):
    cnt = cnt + 1
    mapping[cnt] = dir1

    for file in os.listdir(os.path.join(img_folder, dir1)):

        image_path = os.path.join(img_folder, dir1, file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)  # get the LBP histogram here.
        hist = np.array(hist).reshape(-1, 1)
        hist = hist.T
        labels.append(cnt)
        if i == 0:
            array = np.vstack(hist)
        else:
            array = np.vstack([array, hist])
        i = i + 1

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(array, labels)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        hist = desc.describe(roi_gray)  # get the LBP histogram here.
        hist = np.array(hist).reshape(-1, 1)
        hist = hist.T
        roi_color = img[y:y + h, x:x + w]
        output = clf.predict(hist)
        cv2.putText(img, str(mapping[output[0]]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()











