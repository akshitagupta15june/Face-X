from facerda import FaceRDA
from centerface.centerface import CenterFace
from utils import crop_img, plot_vertices
import cv2


def test_facerda():
    frame = cv2.imread("00.jpg")
    cv2.imshow('frame', frame)

    h, w = frame.shape[:2]
    centerface = CenterFace(h, w, landmarks=True)
    model_path = "../model/frda_sim.onnx"
    facerda = FaceRDA(model_path, True)

    dets, _ = centerface(frame, threshold=0.5)  # 3. forward
    if dets.shape[0] == 0:
        return
    for det in dets:
        boxes, score = det[:4].astype("int32"), det[4]
        roi_box = centerface.get_crop_box(boxes[0], boxes[1], boxes[2] - boxes[0], boxes[3] - boxes[1], 1.4)
        face, ret_roi = crop_img(frame, roi_box)
        vertices = facerda(face, roi_box)
        frame = plot_vertices(frame, vertices)

    cv2.imshow('image', frame)
    cv2.waitKey(0)


def camera_facerda():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()

    h, w = frame.shape[:2]
    centerface = CenterFace(h, w, landmarks=True)
    model_path = "../model/frda_sim.onnx"
    facerda = FaceRDA(model_path, True)

    while success:
        success, frame = cap.read()
        dets, _ = centerface(frame, threshold=0.5)  # 3. forward
        if dets.shape[0] == 0:
            continue
        for det in dets:
            boxes, score = det[:4].astype("int32"), det[4]
            roi_box = centerface.get_crop_box(boxes[0], boxes[1], boxes[2] - boxes[0], boxes[3] - boxes[1], 1.4)
            face, ret_roi = crop_img(frame, roi_box)
            vertices = facerda(face, roi_box)
            frame = plot_vertices(frame, vertices)

        cv2.imshow('frame', frame)
        cv2.waitKey(30)


if __name__ == "__main__":
    # test_facerda()
    camera_facerda()
