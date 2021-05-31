import cv2

def main():
    """
    The function adjusts the background and foreground to form the resultant
    desired filter
    """
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            wanted_img = cv2.imread("wanted.png", -1)

            # selecting all 3 color channels in the frame
            frame_mask_wanted = wanted_img[:, :, 3]

            frame_mask_inv_wanted = cv2.bitwise_not(frame_mask_wanted)
            wanted_img = wanted_img[:, :, 0:3]

            # Dimensions of actual wanted
            ogwantedHt, ogwantedWd  = wanted_img.shape[:2]

            wantedWd = int(frame_width)
            wantedHt = int(frame_height)

            wanted = cv2.resize(wanted_img, (wantedWd, wantedHt), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(frame_mask_wanted, (wantedWd, wantedHt), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(frame_mask_inv_wanted, (wantedWd, wantedHt), interpolation=cv2.INTER_AREA)

            # Obtaining the region-of-interest
            y1 = 0
            y2 = int(y1 + wantedHt)
            x1 = 0
            x2 = int(x1 + wantedWd)
            roi1 = frame[y1:y2, x1:x2]
            roi_bg = cv2.bitwise_and(roi1, roi1, mask=mask_inv)
            roi_fg = cv2.bitwise_and(wanted, wanted, mask=mask)

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

