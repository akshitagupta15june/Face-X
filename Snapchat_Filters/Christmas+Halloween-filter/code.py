import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# photo on which filter will be applied


myphoto = cv2.imread('WIN_20211207_13_35_00_Pro.jpg')  # photo on which filter will be applied
# converting into gray for better resolution
gray = cv2.cvtColor(myphoto, cv2.COLOR_BGR2GRAY)


#    multiple face detection
fl = face.detectMultiScale(gray, 1.09, 7)
ey = face.detectMultiScale(gray, 1.09, 7)
ep = face.detectMultiScale(gray, 1.09, 7)


#    filters
teeth = cv2.imread("teeth.png")
hat = cv2.imread('chrimahat.png')
glass = cv2.imread('ween.png')


# function for placing filters
def Place_Hat(hat, fc, x, y, w, h):
    face_width = w
    face_height = h
    hat_width = face_width + 1
    hat_height = int(face_height) + 1
    hat = cv2.resize(hat, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k] < 235:
                    fc[y + i - int(0.9*face_height)][x + j][k] = hat[i][j][k]
    return fc


def Place_Glass(glass, fc, x, y, w, h):
    face_width = w
    face_height = h

    glass_width = face_width + 1
    glass_height = int(0.50 * face_height) + 1

    glass = cv2.resize(glass, (glass_width, glass_height))

    for i in range(glass_height):
        for j in range(glass_width):
            for k in range(3):
                if glass[i][j][k] < 235:
                    fc[y + i - int(-0.10 * face_height)][x +
                                                         j][k] = glass[i][j][k]
    return fc


def Place_Teeth(teeth, fc, x, y, w, h):
    face_width = w
    face_height = h

    teeth_width = face_width + 1
    teeth_height = int(0.550 * face_height) + 1

    teeth = cv2.resize(teeth, (teeth_width, teeth_height))

    for i in range(teeth_height):
        for j in range(teeth_width):
            for k in range(3):
                if teeth[i][j][k] < 235:
                    fc[y + i - int(-0.550 * face_height)
                       ][x + j][k] = teeth[i][j][k]
    return fc


for (x, y, w, h) in fl:
    frame = Place_Hat(hat, myphoto, x, y, w, h)


for (x, y, w, h) in ey:
    frame = Place_Glass(glass, myphoto, x, y, w, h)


for (x, y, w, h) in ep:
    frame = Place_Teeth(teeth, myphoto, x, y, w, h)



cv2.imshow('Halloween+christmas', frame)
cv2.waitKey(15000) & 0xff
cv2.destroyAllWindows()
