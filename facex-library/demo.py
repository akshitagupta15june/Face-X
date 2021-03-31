from facex import cartoonify, face_detection, blur_bg, ghost_img, mosaic, sketch
import cv2

## Cartoonify effect(Similar with face_det, blur_bg, ghost_img)
image = sketch(img_path='face.jpg')
cv2.imshow("cartoon", image)
cv2.waitKey()


## Mosaic Effect
image = mosaic(img_path='face.jpg', x=219, y=61, w=460-219, h=412-61)
cv2.imshow("ghost", image)
cv2.waitKey()
