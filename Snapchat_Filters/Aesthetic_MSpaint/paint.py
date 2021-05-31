import cv2

def main():
    """
    We place our foreground in such a way that it fits the screen 
    and get results accordingly.
    """

    # Getting video feed through webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            Paint_img = cv2.imread("paint.png", -1)

            # selecting all 3 color channels in the frame
            frame_mask_Paint = Paint_img[:, :, 3]

            frame_mask_inv_Paint = cv2.bitwise_not(frame_mask_Paint)
            Paint_img = Paint_img[:, :, 0:3]

            # Dimensions of actual Paint
            ogPaintHt, ogPaintWd  = Paint_img.shape[:2]

            PaintWd = int(frame_width)
            PaintHt = int(frame_height)

            Paint = cv2.resize(Paint_img, (PaintWd, PaintHt), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(frame_mask_Paint, (PaintWd, PaintHt), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(frame_mask_inv_Paint, (PaintWd, PaintHt), interpolation=cv2.INTER_AREA)

            # Obtaining the region-of-interest
            y1 = 0
            y2 = int(y1 + PaintHt)
            x1 = 0
            x2 = int(x1 + PaintWd)
            roi1 = frame[y1:y2, x1:x2]
            roi_Paint = cv2.bitwise_and(roi1, roi1, mask=mask_inv)
            roi_fg = cv2.bitwise_and(Paint, Paint, mask=mask)

            # Adding filter to the frame
            frame[y1:y2, x1:x2] = cv2.add(roi_Paint, roi_fg)

            cv2.imshow('MS PAINT', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

