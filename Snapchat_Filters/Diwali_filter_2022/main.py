import cv2
import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
cap =cv2.VideoCapture(0)
#  set window size 500x500
cap.set(3,500)
cap.set(4,500)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
imgBG = cv2.imread("Diwali_filter_2022\images\DiwaliBg.png")
face_cascade = cv2.CascadeClassifier('Diwali_filter_2022\haarcascade_frontalface_default.xml')

specs_ori = cv2.imread('Diwali_filter_2022\images\glasses.png', -1)

# Camera Init
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FPS, 30)

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  
    rows, cols, _ = src.shape 
    y, x = pos[0], pos[1] 

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,imgBG,threshold=0.8)

    faces = face_cascade.detectMultiScale(imgOut, 1.2, 5, 0, (120, 120), (350, 350))
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            glass = int(y + 1.5 * h / 5)
            glass1 = int(y + 2.5 * h / 5)
            sh_glass = glass1 - glass

            face_glass_roi_color = imgOut[glass:glass1, x:x + w]

            specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
            transparentOverlay(face_glass_roi_color, specs)
    imgstack =cvzone.stackImages([img,imgOut],2,1)
    cv2.imshow('Diwali Filter 2022', imgstack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
