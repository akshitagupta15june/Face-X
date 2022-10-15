'''import cv2
import cvzone  
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0);
cap.set(3, 640)
cap.set(4, 480)  
cap.set(cv2.CAP_PROP_FPS,60 )
segmentor = SelfiSegmentation() 
fpsreader = cvzone.FPS()
imgBg = cv2.imread("Images/1.jpg")

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,(imgBg), threshold= 0.8  )
    imgstacked = fpsreader.update(imgstacked)

    imgstacked = cvzone.stackImages([img, imgOut],2,1)
    cvzone.imshow("Image",imgstacked)
    cv2.imwrite("Image",img)
    cv2.waitKey(1) '''
#The best alternative code for surrounding a image with background.
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()

# read image
imgOffice = cv2.imread('Images/2.jpg')

#resize office to 640x480
imgOffice = cv2.resize(imgOffice, (640, 480))

imgback = cv2.imread('Images/1.jpg')

imgback = cv2.resize(imgback, (640,480))

imgNoBg = segmentor.removeBG(imgOffice,imgback, threshold=0.50)

# show both images
cv2.imshow('Face',imgOffice)
cv2.imshow('Backgroundsurrounded',imgNoBg)


cv2.waitKey(0)
cv2.destroyAllWindows()
