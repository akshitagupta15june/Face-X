import cv2
import matplotlib.pyplot as plt

face = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')



filename=input("Enter image path here:")  #r'C:\Users\xyz\OneDrive\Desktop\images\photo.JPG'
img=cv2.imread(filename)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
fl=face.detectMultiScale(gray,1.09,9)
ey=face.detectMultiScale(gray,1.09,9)
ms=face.detectMultiScale(gray,1.09,9)
hat=cv2.imread('hat.png')
glass=cv2.imread('pirates.png')
mst=cv2.imread('moustache.png')

def put_hat(hat, fc, x, y, w, h):
    face_width = w
    face_height = h
    hat_width = face_width + 1
    hat_height = int(0.65 * face_height) + 1
    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k] < 235:
                    fc[y + i - int(0.50 * face_height)][x + j][k] = hat[i][j][k]
    return fc


def put_glass(glass, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int(0.50 * face_height) + 1

    glass = cv2.resize(glass, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if glass[i][j][k] < 235:
                    fc[y + i - int(-0.20 * face_height)][x + j][k] = glass[i][j][k]
    return fc

def put_moustache(mst,fc,x,y,w,h):
    
    face_width = w
    face_height = h

    mst_width = int(face_width*0.4166666)+15
    mst_height = int(face_height*0.142857)+30



    mst = cv2.resize(mst,(mst_width,mst_height))

    for i in range(int(0.62857142857*face_height),int(0.62857142857*face_height)+mst_height):
        for j in range(int(0.29166666666*face_width),int(0.29166666666*face_width)+mst_width):
            for k in range(3):
                if mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k] <235:
                    fc[y+i][x+j][k] = mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k]
    return fc


for (x, y, w, h) in ms:
    frame=put_moustache(mst,img, x, y, w, h)
    

    
for (x, y, w, h) in fl:
    frame = put_hat(hat, img, x, y, w, h)
for (x, y, w, h) in ey:
    frame=put_glass(glass,img, x, y, w, h)
    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow('image',frame)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(5)
plt.title("Pirates Filter")
plt.imshow(frame)

plt.show()


cv2.waitKey(8000)& 0xff
cv2.destroyAllWindows()
