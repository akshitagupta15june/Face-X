import cv2
import dlib

def filter(frame, landmarks):
    micyimg= cv2.imread("download.png", -1)      #This line of code just fetch the image which is present in our directory 
    micyMask = micyimg[:, :, 3]                 # it convert image into binary image
    micyMaskInv = cv2.bitwise_not(micyMask)          # inverting the binary image
    micyimg = micyimg[:, :, 0:3]

    # Dimensions of the micy
    micyHt, micyWd = micyimg.shape[:2]

    # Calculate the width and height of the bounding box
    face_width = landmarks.part(16).x - landmarks.part(0).x
    face_height = landmarks.part(8).y - landmarks.part(19).y

    # Increase the size of the bounding box by 25% or less
    width_increase = min(int(face_width * 0.60), int((frame.shape[1] - face_width) / 2))
    height_increase = min(int(face_height * 0.75), int((frame.shape[0] - face_height) / 2))

    x1 = landmarks.part(0).x - width_increase
    y1 = landmarks.part(20).y - height_increase
    x2 = landmarks.part(16).x + width_increase
    y2 = landmarks.part(14).y + height_increase

    # Ensure the ROI coordinates are within the frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    # Calculate the new width and height of the ROI
    roi_width = x2 - x1
    roi_height = y2 - y1

   # These lines resize the Demon image and its mask to match the size of the new bounding box
    demon = cv2.resize(micyimg, (roi_width, roi_height), cv2.INTER_AREA)
    mask = cv2.resize(micyMask, (roi_width, roi_height), cv2.INTER_AREA)
    mask_inv = cv2.resize(micyMaskInv, (roi_width, roi_height), cv2.INTER_AREA)

   
    
    #This line extracts the region of interest (ROI) from the frame using the calculated bounding box coordinates.
    roi = frame[y1:y2, x1:x2]   
    backGround = cv2.bitwise_and(roi, roi, mask=mask_inv).astype('uint8')
    foreGround = cv2.bitwise_and(demon,demon, mask=mask).astype('uint8')

    # Add the filter to the frame
    frame[y1:y2, x1:x2] = cv2.add(backGround, foreGround)   

    return frame


# Declaring the detector
detector = dlib.get_frontal_face_detector()
# Locating the facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        faces = detector(gray)   #This line uses the face detector to detect faces in the grayscale frame.

        # Going through all detected faces
        for face in faces:
            landmarks = predictor(gray, face)
            frame = filter(frame, landmarks)

        cv2.imshow('Demon Filter', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()