import cv2
from PIL import Image
import numpy as np

save = True

def show(image):
    global save

    #open transparent filter
    fMask = Image.open('devilFilter.png')

    #convert colored image into gray scale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #use face.xml to detect face(s)
    faceFile = cv2.CascadeClassifier('face.xml')
    faces = faceFile.detectMultiScale(gray,1.3,5)

    #convert array into image
    bg = Image.fromarray(image)

    #paste filter onto background image
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (y, x), ((y + h), (x + w+70)), (0, 0, 255), 2)
        #resize filter to the size of face detected
        fMask = fMask.resize((w,h))
        #paste filter on background image
        bg.paste(fMask, (x,y-(h//2)), mask=fMask)

    #press 's' key to save your filtered image
    #then only you should press 'esc' key to exit the window
    if save==True and cv2.waitKey(1) == ord('s'):
        cv2.imwrite('savedPicture.jpg', np.asarray(bg))
        save=False

    #return image in array
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
