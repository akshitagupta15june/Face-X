import cv2
import dlib

def filter(frame,landmarks):
    ironManImg = cv2.imread("ironManMask.png",-1)
    ironManMask = ironManImg[:, :, 3] # binary image
    ironManMaskInv = cv2.bitwise_not(ironManMask) # inverting the binary img
    ironManImg = ironManImg[:, :, 0:3]

    # dimensions of the ironMan
    ironManHt, ironManWd  = ironManImg.shape[:2]

    # adjusting dimensions according to the landmarks
    ironManWd1 = abs(landmarks.part(17).x - landmarks.part(26).x) + 75
    ironManHt1 = int(ironManWd1 * ironManHt / ironManWd)
    
    # resize the ironMan img
    ironMan = cv2.resize(ironManImg, (ironManWd1, ironManHt1), cv2.INTER_AREA)
    mask = cv2.resize(ironManMask, (ironManWd1, ironManHt1), cv2.INTER_AREA)
    mask_inv = cv2.resize(ironManMaskInv, (ironManWd1, ironManHt1), cv2.INTER_AREA)

    # grab the region of interest and apply the ironMan
    x1 = int(landmarks.part(27).x - (ironManWd1 / 2))
    y1 = int(landmarks.part(24).y - 100)
    x2 = int(x1 + ironManWd1)
    y2 = int(y1 + ironManHt1)
    
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

        cv2.imshow('Iron man Filter', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
