import cv2
import matplotlib.pyplot as plt

face = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')

filename=input("Enter image path here:")  #r'C:\Users\xyz\OneDrive\Desktop\images\photo.JPG'
img=cv2.imread(filename)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ey=face.detectMultiScale(gray,1.09,9)

butterfly=cv2.imread('Butterfly.png')


def put_butterflyglass(butterfly, fc, x, y, w, h):
    face_width = w
    face_height = h

    butterfly_width = face_width + 1
    butterfly_height = int(0.8 * face_height) + 1

    butterfly = cv2.resize(butterfly, (butterfly_width, butterfly_height))

    for i in range(butterfly_height):
        for j in range(butterfly_width):
            for k in range(3):
                if butterfly[i][j][k] < 235:
                    fc[y + i - int(-0.01 * face_height)][x + j][k] = butterfly[i][j][k]
    return fc

for (x, y, w, h) in ey:
    frame=put_butterflyglass(butterfly,img, x, y, w, h)
       
cv2.imshow('image',frame)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig = plt.figure()
fig.set_figheight(20)
fig.set_figwidth(10)
plt.title("Butterfly Filter")
plt.imshow(frame)

plt.show()


cv2.waitKey(8000)& 0xff
cv2.destroyAllWindows()
