import cv2
import dlib

def filter(frame, landmarks):
    SpidermanImg = cv2.imread("Snapchat_Filters/Spiderman_Filter/spiderman.png", -1)
    SpidermanMask = SpidermanImg[:, :, 3]  # binary image
    SpidermanMaskInv = cv2.bitwise_not(SpidermanMask)  # inverting the binary image
    SpidermanImg = SpidermanImg[:, :, 0:3]

    # Dimensions of the Spiderman
    SpidermanHt, SpidermanWd = SpidermanImg.shape[:2]

    # Calculate the width and height of the bounding box
    face_width = landmarks.part(16).x - landmarks.part(0).x
    face_height = landmarks.part(8).y - landmarks.part(19).y

    # Increase the size of the bounding box by 25% or less
    width_increase = min(int(face_width * 0.20), int((frame.shape[1] - face_width) / 2))
    height_increase = min(int(face_height * 0.70), int((frame.shape[0] - face_height) / 2))

    x1 = landmarks.part(0).x - width_increase
    y1 = landmarks.part(19).y - height_increase
    x2 = landmarks.part(16).x + width_increase
    y2 = landmarks.part(15).y + height_increase

    # Ensure the ROI coordinates are within the frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    # Calculate the new width and height of the ROI
    roi_width = x2 - x1
    roi_height = y2 - y1

    # Resize the Spiderman image and mask to match the new ROI size
    Spiderman = cv2.resize(SpidermanImg, (roi_width, roi_height), cv2.INTER_AREA)
    mask = cv2.resize(SpidermanMask, (roi_width, roi_height), cv2.INTER_AREA)
    mask_inv = cv2.resize(SpidermanMaskInv, (roi_width, roi_height), cv2.INTER_AREA)

    # Grab the region of interest and apply the Spiderman
    roi = frame[y1:y2, x1:x2]
    backGround = cv2.bitwise_and(roi, roi, mask=mask_inv).astype('uint8')
    foreGround = cv2.bitwise_and(Spiderman, Spiderman, mask=mask).astype('uint8')

    # Add the filter to the frame
    frame[y1:y2, x1:x2] = cv2.add(backGround, foreGround)

    return frame


# Declaring the detector
detector = dlib.get_frontal_face_detector()
# Locating the facial landmarks
predictor = dlib.shape_predictor('Snapchat_Filters/Spiderman_Filter/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces in the frame
        faces = detector(gray)

        # Going through all detected faces
        for face in faces:
            landmarks = predictor(gray, face)
            frame = filter(frame, landmarks)

        cv2.imshow('Spiderman Filter', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
