import numpy as np
import cv2
import random

# Load Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load images for glasses, cigar, mustache, and butterfly
specs_ori = cv2.imread('glass3.png', -1)
cigar_ori = cv2.imread('cigar.png', -1)
mus_ori = cv2.imread('m1.png', -1)
butterfly_ori = cv2.imread('butter3.png', -1)  # Make sure the butterfly image has a transparent background

# Resize butterfly image to a random scale between 0.2 and 0.5 of its original size
butterfly_scale = random.uniform(0.2, 0.5)
butterfly_height, butterfly_width, _ = butterfly_ori.shape
butterfly_resized = cv2.resize(butterfly_ori, (int(butterfly_width * butterfly_scale),
                                               int(butterfly_height * butterfly_scale)))

# Camera Init
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


def transparentOverlay(src, overlay, pos=(0, 0)):
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # Loop through the pixels of the overlay image and blend them with the background image using alpha channel
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel (transparency)
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using the Haar Cascade Classifier
    faces = face_cascade.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350))

    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            # Calculate the regions for glasses, cigar, and mustache placement
            glass_symin = int(y + 1.5 * h / 5)
            glass_symax = int(y + 2.5 * h / 5)
            sh_glass = glass_symax - glass_symin

            cigar_symin = int(y + 4 * h / 6)
            cigar_symax = int(y + 5.5 * h / 6)
            sh_cigar = cigar_symax - cigar_symin

            mus_symin = int(y + 3.5 * h / 6)
            mus_symax = int(y + 5 * h / 6)
            sh_mus = mus_symax - mus_symin

            # Extract regions of interest for glasses, cigar, and mustache from the face
            face_glass_roi_color = img[glass_symin:glass_symax, x:x + w]
            face_cigar_roi_color = img[cigar_symin:cigar_symax, x:x + w]
            face_mus_roi_color = img[mus_symin:mus_symax, x:x + w]

            # Resize glasses, cigar, and mustache to fit the face regions
            specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
            # cigar = cv2.resize(cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_CUBIC)
            mustache = cv2.resize(mus_ori, (w, sh_mus), interpolation=cv2.INTER_CUBIC)

            # Overlay glasses, cigar, and mustache on the face
            transparentOverlay(face_glass_roi_color, specs)
            # transparentOverlay(face_cigar_roi_color, cigar, (int(w / 2), int(sh_cigar / 2)))
            transparentOverlay(face_mus_roi_color, mustache)

    # Overlay 10-15 butterfly images randomly throughout the frame
    num_butterflies = random.randint(10, 15)
    for _ in range(num_butterflies):
        bh, bw, _ = butterfly_resized.shape
        rand_x = random.randint(0, img.shape[1] - bw)
        rand_y = random.randint(0, img.shape[0] - bh)
        transparentOverlay(img, butterfly_resized, pos=(rand_y, rand_x))

    cv2.imshow('anamika', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('img.jpg', img)
        break

cap.release()
cv2.destroyAllWindows()
