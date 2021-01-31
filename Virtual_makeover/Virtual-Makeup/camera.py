import cv2
import dlib
import numpy as np


camera = True
cap = cv2.VideoCapture(0)

def empty(a):
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",640,240)
cv2.createTrackbar("Blue","BGR",0,255,empty)
cv2.createTrackbar("Green","BGR",0,255,empty)
cv2.createTrackbar("Red","BGR",0,255,empty)
def create(img, points,masked = False, cropped = True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points],(255,255,255))
        # cv2.imshow("mask",mask)
        img = cv2.bitwise_and(img,mask)
    if cropped:
        b = cv2.boundingRect(points)
        x,y,w,h = b
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,5,5)
        return imgCrop
    else:
        return mask
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    if camera: success , img = cap.read()
    else: print("Camera not working")
    img = cv2.resize(img,(0,0), None,1,1)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        # imgOri = cv2.rectangle(imgOriginal,(x1,y1),(x2,y2),(0,255,0),1)
        landmarks = predictor(imgGray,face)
        mypoints = []
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mypoints.append([x,y])
            # cv2.circle(imgOriginal,(x,y),2,(0,0,255),3)
            # cv2.putText(imgOriginal,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        mypoints = np.array(mypoints)
        lips = create(img,mypoints[48:61],masked=True,cropped=False)
        # cv2.imshow("Lip",lips)
        imgColor = np.zeros_like(lips)
        a = cv2.getTrackbarPos("Blue","BGR")
        q = cv2.getTrackbarPos("Green","BGR")
        w = cv2.getTrackbarPos("Red","BGR")
        imgColor[:] = a,q,w
        # cv2.imshow("Color",imgColor)
        imgColor = cv2.bitwise_and(lips,imgColor)
        imgColor = cv2.GaussianBlur(imgColor,(7,7),10)
        # imgOriginal_Image = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)
        # imgOriginal_Image = cv2.cvtColor(imgOriginal_Image,cv2.COLOR_GRAY2BGR)
        imgColor =cv2.addWeighted(imgOriginal,1,imgColor,0.4,0)
        cv2.imshow("BGR",imgColor)
    # cv2.imshow("Image",imgOriginal)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
