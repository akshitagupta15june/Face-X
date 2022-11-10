import cv2
import dlib

def filter(frame,landmarks):
    HpumpkinImg = cv2.imread("pumpkin.png",-1)
    HpumpkinMask = HpumpkinImg[:, :, 3] # binary image
    HpumpkinMaskInv = cv2.bitwise_not(HpumpkinMask) # inverting the binary img
    HpumpkinImg = HpumpkinImg[:, :, 0:3]

    # dimensions of the Hpumpkin
    HpumpkinHt, HpumpkinWd  = HpumpkinImg.shape[:2]

    # adjusting dimensions according to the landmarks
    HpumpkinWd1 = abs(landmarks.part(18).x - landmarks.part(26).x) + 75
    HpumpkinHt1 = int(HpumpkinWd1 * HpumpkinHt / HpumpkinWd)
    
    # resize the Hpumpkin img
    Hpumpkin = cv2.resize(HpumpkinImg, (HpumpkinWd1, HpumpkinHt1), cv2.INTER_AREA)
    mask = cv2.resize(HpumpkinMask, (HpumpkinWd1, HpumpkinHt1), cv2.INTER_AREA)
    mask_inv = cv2.resize(HpumpkinMaskInv, (HpumpkinWd1, HpumpkinHt1), cv2.INTER_AREA)

    # grab the region of interest and apply the Hpumpkin
    x1 = int(landmarks.part(27).x - (HpumpkinWd1 / 2))
    y1 = int(landmarks.part(16).y - 100)
    x2 = int(x1 + HpumpkinWd1)
    y2 = int(y1 + HpumpkinHt1)
    
    roi = frame[y1:y2, x1:x2]
    backGround = cv2.bitwise_and(roi, roi, mask=mask_inv).astype('uint8')
    foreGround = cv2.bitwise_and(Hpumpkin, Hpumpkin, mask=mask).astype('uint8')
    
    # Adding filter to the frame
    frame[y1:y2, x1:x2] = cv2.add(backGround, foreGround)

    return frame


#declaring the detector
detector = dlib.get_frontal_face_detector()
# locating the facial landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecting faces in the frame
        faces = detector(gray)

        # going through all detected faces
        for face in faces:
            landmarks = predictor(gray, face)
            frame = filter(frame, landmarks)

        cv2.imshow('Pumpkin Filter', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
