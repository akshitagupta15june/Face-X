import cv2
import matplotlib.pyplot as plt

face = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')



filename=input("Enter your image path here:")  #ex -> r'C:\Users\xyz\images\p1.jpg' 
img=cv2.imread(filename)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
fl=face.detectMultiScale(gray,1.09,9)
ey=face.detectMultiScale(gray,1.09,9)
mantika_png=input("Enter the given fore head jewellery image path here:")  #ex -> r'C:\Users\xyz\images\fore_head_jewellery.jpg'  
earing_png=input("Enter the given earing jewellery image path here:")  #ex -> r'C:\Users\xyz\images\earing.jpg' 
mantika=cv2.imread(mantika_png)
earing=cv2.imread(earing_png)


def put_mantika(mantika, fc, x, y, w, h):
    face_width = w
    face_height = h
    mantika_width = face_width + 1
    mantika_height = int(0.8 * face_height) + 1
    mantika = cv2.resize(mantika, (mantika_width, mantika_height))

    for i in range(mantika_height):
        for j in range(mantika_width):
            for k in range(3):
                if mantika[i][j][k] < 235:
                    fc[y + i - int(0.25 * face_height)][x + j][k] = mantika[i][j][k]
    return fc


def put_earing(earing, fc, x, y, w, h):
    face_width = w
    face_height = h

    earing_width = face_width + 7
    earing_height = int(1 * face_height) + 1

    earing = cv2.resize(earing, (earing_width, earing_height))

    for i in range(earing_height):
        for j in range(earing_width):
            for k in range(3):
                if earing[i][j][k] < 235:
                    fc[y + i - int(-0.6 * face_height)][x + j][k] = earing[i][j][k]
    return fc

for (x, y, w, h) in ey:
    frame=put_earing(earing,img, x, y, w, h)

for (x, y, w, h) in fl:
    frame = put_mantika(mantika, img, x, y, w, h)

    
    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow('image',frame)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig = plt.figure()
fig.set_figheight(30)
fig.set_figwidth(15)
plt.title("Jewellery Filter")
plt.imshow(frame)

plt.show()
