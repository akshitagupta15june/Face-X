import cv2


def main():

    # Getting video feed through webcam
    cap = cv2.VideoCapture(0)
    while True:
        getFrame, frame = cap.read()
        if getFrame:
            # Getting the dimensions of the feed through the webcam
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            img = cv2.imread("Happy_Holi_Filter.jpg", -1)

            # Selecting all the 3 color channels in the frame
            frame_mask = img[:, :, 2]
            frame_mask_inv = cv2.bitwise_not(frame_mask)
            mask_img = img[:, :, 0:3]

            # Redefining the dimensions
            wd = int(width)
            ht = int(height)

            # Using the redefined dimensions
            holi = cv2.resize(mask_img, (wd, ht))
            mask = cv2.resize(frame_mask, (wd, ht))
            mask_inv = cv2.resize(frame_mask_inv, (wd, ht))

            # Obtaining the region-of-interest
            y1 = 0
            y2 = int(y1 + ht)
            x1 = 0
            x2 = int(x1 + wd)
            frame0 = frame[y1:y2, x1:x2]
            frame1 = cv2.bitwise_and(frame0, frame0, mask=mask_inv)
            frame2 = cv2.bitwise_and(holi, holi, mask=mask)

            # Adding filter to the frame
            frame[y1:y2, x1:x2] = cv2.add(frame1, frame2)

            cv2.imshow('Snapchat Filter Holi', frame)

            cv2.resizeWindow('Snapchat Filter Holi', 640, 481)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()