import cv2

def main():
    """
    Here we arrange the background and foreground accordingly to get desired results
    """

    # Getting video feed through webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            Mag_img = cv2.imread("time.png", -1)

            # selecting all 3 color channels in the frame
            frame_mask_Mag = Mag_img[:, :, 3]

            frame_mask_inv_Mag = cv2.bitwise_not(frame_mask_Mag)
            Mag_img = Mag_img[:, :, 0:3]

            # Dimensions of actual Mag
            ogMagHt, ogMagWd  = Mag_img.shape[:2]

            MagWd = int(frame_width)
            MagHt = int(frame_height)

            Mag = cv2.resize(Mag_img, (MagWd, MagHt), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(frame_mask_Mag, (MagWd, MagHt), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(frame_mask_inv_Mag, (MagWd, MagHt), interpolation=cv2.INTER_AREA)

            # Obtaining the region-of-interest
            y1 = 0
            y2 = int(y1 + MagHt)
            x1 = 0
            x2 = int(x1 + MagWd)
            roi1 = frame[y1:y2, x1:x2]
            roi_bg = cv2.bitwise_and(roi1, roi1, mask=mask_inv)
            roi_fg = cv2.bitwise_and(Mag, Mag, mask=mask)

            # Adding filter to the frame
            frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

            cv2.imshow('Time', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

