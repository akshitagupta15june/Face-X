import dlib
import cv2

# Load the face detector model
detector = dlib.get_frontal_face_detector()

# Load the landmarks predictor model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the Ghost Rider mask image with alpha channel
mask = cv2.imread("ghost_rider_mask.png", -1)

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Detect landmarks for the current face
        landmarks = predictor(gray, face)

        # Get the coordinates of the eyes and mouth
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
        mouth_right = (landmarks.part(54).x, landmarks.part(54).y)

        # Calculate the width and height of the face
        face_width = int((right_eye[0] - left_eye[0]) * 1.5)
        face_height = int((mouth_left[1] - left_eye[1]) * 1.5)

        # Resize the Ghost Rider mask to match the face size
        mask_resized = cv2.resize(mask, (face_width, face_height))

        # Calculate the position of the mask
        x1 = int(left_eye[0] - face_width * 0.25)
        x2 = int(x1 + face_width)
        y1 = int(left_eye[1] - face_height * 0.5)
        y2 = int(y1 + face_height)

        # Calculate the region of interest (ROI) for the mask
        roi = frame[y1:y2, x1:x2]

        # Create a mask from the mask image's alpha channel
        mask_alpha = mask_resized[:, :, 3] / 255.0

        # Invert the mask alpha channel
        mask_alpha_inv = 1.0 - mask_alpha

        # Extract the masked region from the mask image (excluding the alpha channel)
        mask_rgb = mask_resized[:, :, 0:3]

        # Multiply the mask region with the mask alpha channel
        masked_roi = cv2.multiply(mask_alpha, mask_rgb.astype(float))

        # Multiply the original region with the inverted mask alpha channel
        roi_inv = cv2.multiply(mask_alpha_inv, roi.astype(float))

        # Add the masked region and the original region
        masked_roi = masked_roi.astype(np.uint8)
        roi_inv = roi_inv.astype(np.uint8)
        face_roi = cv2.add(masked_roi, roi_inv)

        # Replace the face region with the masked face region
        frame[y1:y2, x1:x2] = face_roi

    # Display the frame
    cv2.imshow("Ghost Rider Filter", frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
