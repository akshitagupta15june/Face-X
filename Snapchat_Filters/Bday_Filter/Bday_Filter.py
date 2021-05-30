import dlib
import cv2


def mask(frame, landmarks):
    imgHorn = cv2.imread("horn.png", -1)
    orig_mask = imgHorn[:, :, 3]
    orig_mask_inv = cv2.bitwise_not(orig_mask)
    imgHorn = imgHorn[:, :, 0:3]
    origHornHeight, origHornWidth = imgHorn.shape[:2]

    imgHat = cv2.imread("hat.png", -1)
    orig_mask_h = imgHat[:, :, 3]
    orig_mask_inv_h = cv2.bitwise_not(orig_mask_h)
    imgHat = imgHat[:, :, 0:3]
    origHatHeight, origHatWidth = imgHat.shape[:2]

    hornWidth = abs(3 * (landmarks.part(63).x - landmarks.part(61).x) + 20)
    hornHeight = int(hornWidth * origHornHeight / origHornWidth) + 10
    horn = cv2.resize(imgHorn, (hornWidth, hornHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (hornWidth, hornHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (hornWidth, hornHeight), interpolation=cv2.INTER_AREA)
    y1 = int(landmarks.part(62).y - (hornHeight / 2)) + 30
    y2 = int(y1 + hornHeight)
    x1 = int(landmarks.part(62).x - (hornWidth / 2))
    x2 = int(x1 + hornWidth)
    roi = frame[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_fg = cv2.bitwise_and(horn, horn, mask=mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    hatWidth = abs(landmarks.part(26).x - landmarks.part(17).x + 80)
    hatHeight = int(hatWidth * origHatHeight / origHatWidth)
    hat = cv2.resize(imgHat, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask_h, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_h, (hatWidth, hatHeight), interpolation=cv2.INTER_AREA)
    y1 = int(landmarks.part(24).y - 220)
    y2 = int(y1 + hatHeight)
    x1 = int(landmarks.part(27).x - (hatWidth / 2))
    x2 = int(x1 + hatWidth)
    roi1 = frame[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi1, roi1, mask=mask_inv)
    roi_fg = cv2.bitwise_and(hat, hat, mask=mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    return frame


def filter():
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

                frame = mask(frame, landmarks)

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
    filter()
