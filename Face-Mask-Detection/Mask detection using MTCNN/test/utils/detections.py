import cv2
import numpy as np

from PIL import Image
from torch_mtcnn import detect_faces


def fetch_faces(image, return_landmarks=False):
    # standarize detector input from cv2 image to PIL Image
    if isinstance(image, (np.ndarray, np.generic)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    # for some reason, detector randomly throws an error
    try:
        bboxes, landmarks = detect_faces(image)
    except ValueError:
        bboxes, landmarks = [], []

    # postprocess bounding bboxes
    if len(bboxes) > 0:
        scores = bboxes[:, -1]
        bboxes = bboxes[:, :-1].astype("int")
    else:
        scores = []

    return ([bboxes, scores], landmarks) if return_landmarks else bboxes, scores
