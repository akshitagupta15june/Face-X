import dlib
import cv2
from scipy.spatial import distance as dist
import numpy as np


def eye_aspect_ratio(eye):
    """
    It is used to determine EAR value based on the list we passed
    which consists of 6 points.
    :param eye: list of 6 points that we get from landmark
    :return: calculated ear value
    """
    # A & B for Vertical distance
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # C for Horizontal distance
    C = dist.euclidean(eye[0], eye[3])

    # Formula for calculating EAR
    ear = (A + B) / (2.0 * C)

    return ear


def filter_1(frame, landmarks):
    """
    Generating 1 filter according to the face landmarks produced
    from the frame and displaying accordingly.
    :param : the current frame captured, landmarks generated from it
    :return: frame with the filter on it
    """
    # Storing image with the desired filter
    imgGlass = cv2.imread("assets/sun_1.png", -1)

    # selecting all 3 color channels for the image
    orig_mask_g = imgGlass[:, :, 3]
    # Colorwise inverted mask of the image
    orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)
    imgGlass = imgGlass[:, :, 0:3]

    # Generating dimensions of goggles from original image
    origGlassHeight, origGlassWidth = imgGlass.shape[:2]

    # Generating required width and height according to the landmarks
    glassWidth = abs(landmarks.part(16).x - landmarks.part(1).x)
    glassHeight = int(glassWidth * origGlassHeight / origGlassWidth)

    # Resizing the image according to the dimensions generated from the frame
    glass = cv2.resize(imgGlass, (glassWidth, glassHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask_g, (glassWidth, glassHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_g, (glassWidth, glassHeight), interpolation=cv2.INTER_AREA)

    # For obtaining the Region-of-interest (ROI)
    y1 = int(landmarks.part(24).y)
    y2 = int(y1 + glassHeight)
    x1 = int(landmarks.part(27).x - (glassWidth / 2))
    x2 = int(x1 + glassWidth)
    roi1 = frame[y1:y2, x1:x2]

    # Obtaining the background and foreground of the ROI
    roi_bg = cv2.bitwise_and(roi1, roi1, mask=mask_inv)
    roi_fg = cv2.bitwise_and(glass, glass, mask=mask)

    # Adding the filter to the frame
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    return frame


def filter_2(frame, landmarks):
    """
    Generating 2 filter according to the face landmarks produced
    from the frame and displaying accordingly.
    :param : the current frame captured, landmarks generated from it
    :return: frame with the filter on it
    """
    imgMustache = cv2.imread("assets/moustache.png", -1)

    orig_mask = imgMustache[:, :, 3]
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    imgMustache = imgMustache[:, :, 0:3]
    origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

    imgGlass = cv2.imread("assets/glasses.png", -1)
    orig_mask_g = imgGlass[:, :, 3]
    orig_mask_inv_g = cv2.bitwise_not(orig_mask_g)
    imgGlass = imgGlass[:, :, 0:3]
    origGlassHeight, origGlassWidth = imgGlass.shape[:2]

    mustacheWidth = abs(3 * (landmarks.part(31).x - landmarks.part(35).x))
    mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth) - 10
    mustache = cv2.resize(imgMustache, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
    y1 = int(landmarks.part(33).y - (mustacheHeight / 2)) + 10
    y2 = int(y1 + mustacheHeight)
    x1 = int(landmarks.part(51).x - (mustacheWidth / 2))
    x2 = int(x1 + mustacheWidth)
    roi = frame[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_fg = cv2.bitwise_and(mustache, mustache, mask=mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    glassWidth = abs(landmarks.part(16).x - landmarks.part(1).x)
    glassHeight = int(glassWidth * origGlassHeight / origGlassWidth)
    glass = cv2.resize(imgGlass, (glassWidth, glassHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask_g, (glassWidth, glassHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_g, (glassWidth, glassHeight), interpolation=cv2.INTER_AREA)
    y1 = int(landmarks.part(24).y)
    y2 = int(y1 + glassHeight)
    x1 = int(landmarks.part(27).x - (glassWidth / 2))
    x2 = int(x1 + glassWidth)
    roi1 = frame[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi1, roi1, mask=mask_inv)
    roi_fg = cv2.bitwise_and(glass, glass, mask=mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    return frame


def snapchat_filter():
    """
    This function consists main logic of the program in which
    1. detect faces
    2. from 68 landmark points we detect eyes
    3. from that points, calculation eye aspect ratio (EAR), then taking
       median of both eye EAR ratios.
    4. Checking for how many frames EAR is below our Threshold limit indicating,
       closed eyes.
    5. if eyes closed for more than the threshold we set for frames means person
       is feeling drowsy.
    :return: None
    """

    # detector for detecting the face in the image
    detector = dlib.get_frontal_face_detector()
    # predictor of locating 68 landmark points from the face by using a pretrained model
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    EYE_AR_THRESH = 0.27
    # if eye is closed (ear < threshold) for a minimum consecutive frames ie person
    # feeling drowsy.
    EYE_AR_CONSEC_FRAMES = 5
    # for keeping count of frames below ear
    COUNTER = 0
    Blink = True

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detecting faces in the frame
            faces = detector(frameGray)

            # if faces are present then locating the landmark points
            for face in faces:
                landmarks = predictor(frameGray, face)
                # list for storing points location in pixel.
                landmark_points_location = []

                for i in range(36, 48):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    # calculating x and y and appending it into a list
                    landmark_points_location.append([x, y])

                # changing the list into numpy array to perform computations.
                landmark_points_location = np.array(landmark_points_location)

                leftEye = landmark_points_location[:6]
                rightEye = landmark_points_location[6:]

                # calculating left and right eye EAR
                leftEye_ear = eye_aspect_ratio(leftEye)
                rightEye_ear = eye_aspect_ratio(rightEye)

                # calculating mean EAR
                ear = (leftEye_ear + rightEye_ear) / 2

                # cv2.putText(frame, "EAR: {:.2f}".format(ear), (int(cap.get(3)) - 125, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    # If counter is greater than threshold then DROWSINESS
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if Blink:
                            Blink = False
                        else:
                            Blink = True
                else:
                    if Blink:
                        frame = filter_2(frame, landmarks)
                    else:
                        frame = filter_1(frame, landmarks)

            # for showing frames on the window named Detector
            cv2.imshow('Detector', frame)

            # for quiting the program press 'ESC'
            if cv2.waitKey(1) & 0xFF == 27:
                break

        else:
            break

    # releasing all the frames we captured and destroying the windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    snapchat_filter()
