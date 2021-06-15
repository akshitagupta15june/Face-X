import cv2
import dlib

def filter(frame,landmarks):
    hudImg = cv2.imread("hud.png",-1)
    hudMask = hudImg[:, :, 3] # binary image
    hudMaskInv = cv2.bitwise_not(hudMask) # inverting the binary img
    hudImg = hudImg[:, :, 0:3]

    # dimensions of the hud
    hudHt, hudWd  = hudImg.shape[:2]

    # adjusting dimensions according to the landmarks
    hudWd1 = abs(landmarks.part(17).x - landmarks.part(26).x) + 125
    hudHt1 = int(hudWd1 * hudHt / hudWd)
    
    # resize the hud img
    hud = cv2.resize(hudImg, (hudWd1, hudHt1), cv2.INTER_AREA)
    mask = cv2.resize(hudMask, (hudWd1, hudHt1), cv2.INTER_AREA)
    mask_inv = cv2.resize(hudMaskInv, (hudWd1, hudHt1), cv2.INTER_AREA)

    # grab the region of interest and apply the hud
    x1 = int(landmarks.part(27).x - (hudWd1 / 2))
    y1 = int(landmarks.part(24).y - 30)
    x2 = int(x1 + hudWd1)
    y2 = int(y1 + hudHt1)
    
    roi = frame[y1:y2, x1:x2]
    backGround = cv2.bitwise_and(roi, roi, mask=mask_inv).astype('uint8')
    foreGround = cv2.bitwise_and(hud, hud, mask=mask).astype('uint8')
    
    # Adding filter to the frame
    frame[y1:y2, x1:x2] = cv2.add(backGround, foreGround)
    cv2.imshow('hud view',frame[y1:y2, x1:x2])
    return frame


#declaring the detector
detector = dlib.get_frontal_face_detector()
# locating the facial landmarks
predictor = dlib.shape_predictor('shape_predictor_70_face_landmarks.dat')

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

        cv2.imshow('HuD view', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()