import cv2
import dlib

def filter(frame,landmarks):
    captainAmericaImg = cv2.imread("captainAmericaMask.png",-1)
    captainAmericaMask = captainAmericaImg[:, :, 3] # binary image
    captainAmericaMaskInv = cv2.bitwise_not(captainAmericaMask) # inverting the binary img
    captainAmericaImg = captainAmericaImg[:, :, 0:3]

    # dimensions of the ironMan
    capHt, capWd  = captainAmericaImg.shape[:2]

    # adjusting dimensions according to the landmarks
    capWd1 = abs(landmarks.part(17).x - landmarks.part(26).x) + 75
    capHt1 = int(capWd1 * capHt / capWd)
    
    # resize the ironMan img
    ironMan = cv2.resize(captainAmericaImg, (capWd1, capHt1), cv2.INTER_AREA)
    mask = cv2.resize(captainAmericaMask, (capWd1, capHt1), cv2.INTER_AREA)
    mask_inv = cv2.resize(captainAmericaMaskInv, (capWd1, capHt1), cv2.INTER_AREA)

    # grab the region of interest and apply the ironMan
    x1 = int(landmarks.part(27).x - (capWd1 / 2))
    y1 = int(landmarks.part(24).y - 100)
    x2 = int(x1 + capWd1)
    y2 = int(y1 + capHt1)
    
    roi = frame[y1:y2, x1:x2]
    backGround = cv2.bitwise_and(roi, roi, mask=mask_inv).astype('uint8')
    foreGround = cv2.bitwise_and(ironMan, ironMan, mask=mask).astype('uint8')
    
    # Adding filter to the frame
    frame[y1:y2, x1:x2] = cv2.add(backGround, foreGround)

    return frame


#declaring the detector
detector = dlib.get_frontal_face_detector()
# locating the facial landmarks
predictor = dlib.shape_predictor('shape_predictor_70_face_landmarks.dat')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
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

        cv2.imshow('Captain America Filter', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    filter()