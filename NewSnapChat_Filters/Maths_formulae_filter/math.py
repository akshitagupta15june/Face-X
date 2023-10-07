import cv2
from PIL import Image

def main():
    """
    Here we detect face and the landmarks through a live video feed.
    We use these facial landmarks to place the filter accordingly.
    """

    # detecting the face in the feed
    # detector = dlib.get_frontal_face_detector()
    # # locating the facial landmarks
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Getting video feed through webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # print(frame_width, frame_height)

            Crown_img = cv2.imread("mm.png", -1)
            # print(Crown_img.shape)

            # selecting all 3 color channels in the frame
            frame_mask_crown = Crown_img[:, :, 3]

            frame_mask_inv_crown = cv2.bitwise_not(frame_mask_crown)
            Crown_img = Crown_img[:, :, 0:3]

            # Dimensions of actual crown
            ogCrownHt, ogCrownWd  = Crown_img.shape[:2]

            CrownWd = int(frame_width)
            # CrownHt = int(CrownWd * ogCrownHt / ogCrownWd)
            CrownHt = int(frame_height)

            crown = cv2.resize(Crown_img, (CrownWd, CrownHt), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(frame_mask_crown, (CrownWd, CrownHt), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(frame_mask_inv_crown, (CrownWd, CrownHt), interpolation=cv2.INTER_AREA)

            # Obtaining the region-of-interest
            y1 = 0
            y2 = int(y1 + CrownHt)
            x1 = 0
            x2 = int(x1 + CrownWd)
            roi1 = frame[y1:y2, x1:x2]
            roi_bg = cv2.bitwise_and(roi1, roi1, mask=mask_inv)
            roi_fg = cv2.bitwise_and(crown, crown, mask=mask)

            # Adding filter to the frame
            frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

            cv2.imshow('Detector', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
