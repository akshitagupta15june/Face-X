"""Test for video camera.
(No longer used in the tutorial)
With the test frame in focus, hit `q` to exit.
i.e., Typing `q` into your terminal will do nothing.
"""

import cv2


def main():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()