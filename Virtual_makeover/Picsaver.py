import cv2
cap = cv2.VideoCapture(0)
cv2.namedWindow("Picture Saver")
img_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k == ord("q"):
        # q pressed
        break
    elif k == ord("c"):
        # c pressed
        img_name = "image{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("Success")
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
