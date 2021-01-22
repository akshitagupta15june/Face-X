"""Real-time dog filter

Move your face around and a dog filter will be applied to your face,
if it is not out-of-bounds.
With the test frame in focus, hit `q` to exit.
i.e., Typing `q` into your terminal will do nothing.
"""

from step_7_fer import get_image_to_emotion_predictor
import numpy as np
import cv2


def apply_mask(face: np.array, mask: np.array) -> np.array:
    """Add the mask to the provided face."""
    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = face.shape

    # Resize the mask to fit on face
    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    new_mask_shape = (new_mask_w, new_mask_h)
    resized_mask = cv2.resize(mask, new_mask_shape)

    # Add mask to face - ensure mask is centered
    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)
    off_h = int((face_h - new_mask_h) / 2)
    off_w = int((face_w - new_mask_w) / 2)
    face_with_mask[off_h: off_h+new_mask_h, off_w: off_w+new_mask_w][non_white_pixels] = \
         resized_mask[non_white_pixels]

    return face_with_mask


def main():
    cap = cv2.VideoCapture(0)

    # load mask
    mask0 = cv2.imread('assets/dog.png')
    mask1 = cv2.imread('assets/dalmation.png')
    mask2 = cv2.imread('assets/sheepdog.png')
    masks = (mask0, mask1, mask2)

    # get emotion predictor
    predictor = get_image_to_emotion_predictor()

    # initialize front face classifier
    cascade = cv2.CascadeClassifier(
        "assets/haarcascade_frontalface_default.xml")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_h, frame_w, _ = frame.shape

        # Convert to black-and-white
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blackwhite = cv2.equalizeHist(gray)

        # Detect faces
        rects = cascade.detectMultiScale(
            blackwhite, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # Add mask to faces
        for x, y, w, h in rects:
            # crop a frame slightly larger than the face
            y0, y1 = int(y - 0.25*h), int(y + 0.75*h)
            x0, x1 = x, x + w

            # give up if the cropped frame would be out-of-bounds
            if x0 < 0 or y0 < 0:
                continue

            # apply code that inputted into interactive prompt earlier
            mask = masks[predictor(frame[y:y+h, x: x+w])]
            frame[y0: y1, x0: x1] = apply_mask(frame[y0: y1, x0: x1], mask)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
