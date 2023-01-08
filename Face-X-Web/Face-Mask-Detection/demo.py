from facex import cartoonify, face_detection, blur_bg, ghost_img, mosaic, sketch
from facex import face_mask
import cv2

# ## Cartoonify effect(Command similar to face_det, blur_bg, ghost_img)
# image = cartoonify(img_path='face.jpg')
# cv2.imshow("cartoon", cv2.resize(image, (600,600)))
# cv2.waitKey()


# ## Mosaic Effect
# image = mosaic(img_path='face.jpg', x=219, y=61, w=460-219, h=412-61)
# cv2.imshow("ghost", cv2.resize(image, (600,600)))
# cv2.waitKey()


## Face mask detection(Image)
image = face_mask('face.jpg')
cv2.imshow("face_mask", cv2.resize(image, (600,600)))
cv2.waitKey()


## Face mask detection(Video)
# face_mask('your-video.mp4') 
