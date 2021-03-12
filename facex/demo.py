from facex import cartoonify, face_detection, blur_bg
import cv2

image = face_detection('face.jpg', method='opencv')
cv2.imshow("cartoon", image)
cv2.waitKey()