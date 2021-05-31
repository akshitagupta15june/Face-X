import cv2
from PIL import Image
import numpy as np

save = True

def show(image):
    global save

    #open filter to apply
    fMask = Image.open('pic2.png')

    #convert colored image into gray scale image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #detection of face(s)
    faceFile = cv2.CascadeClassifier('face.xml')
    faces = faceFile.detectMultiScale(gray,1.3,5)

    #convert array into image
    bg = Image.fromarray(image)

    #apply filter onto background image
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), ((y + h), (x + w+70)), (0,0,255 ), -1)
        #resize the filter size
        fMask = fMask.resize((w,h-85))
        #paste it
        bg.paste(fMask, (x,y-(h//2)), mask=fMask)
    
    #press 's' key to save image
    #then only press 'esc' key to exit window
    if save==True and cv2.waitKey(1) == ord('s'):
        cv2.imwrite('savedPicture.jpg', np.asarray(bg))
        save=False

    #return as array
    return np.asarray(bg)

#capture live video stream from webcam
video = cv2.VideoCapture(0)

while True:
    flag,img = video.read()
    cv2.imshow('Video',show(img))
  
    #press 'esc' key to exit window
    if cv2.waitKey(1)==27:      #27 is ascii value of esc
        break

print(video)
#release memory
video.release()
cv2.destroyAllWindows()
