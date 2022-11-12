import time
import cv2

img = cv2.imread('test-image.jpg')
width = img.shape[1]
height = img.shape[0]
bicubic = cv2.resize(img,(width*4,height*4))
cv2.imshow('Image',img)
cv2.imshow('BICUBIC',bicubic)

super_res = cv2.dnn_superres.DnnSuperResImpl_create()

start = time.time()
super_res.readModel('LapSRN_x4.pb')
super_res.setModel('lapsrn',4)
lapsrn_image = super_res.upsample(img)
end = time.time()
print('Time taken in seconds by lapsrn', end-start)
cv2.imshow('LAPSRN',lapsrn_image)

start = time.time()
super_res.readModel('EDSR_x4.pb')
super_res.setModel('edsr',4)
edsr_image = super_res.upsample(img)
end = time.time()
print('Time taken in seconds by edsr', end-start)
cv2.imshow('EDSR',edsr_image)

cv2.waitKey(0)
cv2.destroyAllWindows() 