import argparse
import cv2
import logging
import logging.config

from utils.detections import fetch_faces, fetch_centroids
from utils.drawing import draw_tracker, draw_bounding_boxes
from utils.tracking import CentroidTracker


# setup logger
logging.config.fileConfig("log/logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__file__)


def display(frame, tracker, bboxes, scores):
    canvas = frame.copy()
    canvas = draw_tracker(canvas, tracker)
    canvas = draw_bounding_boxes(canvas, bboxes, scores)
    return canvas


def track_video(vcap):
    tracker = CentroidTracker(max_detections=10)
    while True:
        ret, frame = vcap.read()
        key = cv2.waitKey(50)
        if not ret or key == ord("q"):
            logger.info("Stopped video processing")
            break

        # do face tracking and ignore facial landmarks
        bboxes, scores = fetch_faces(frame, return_landmarks=False)
        # update face tracker
        centroids = fetch_centroids(bboxes)
        tracker.update(centroids)

        # show results
        if len(centroids) > 0:
            frame = display(frame, tracker, bboxes, scores)

        cv2.imshow("", frame)
    cv2.destroyAllWindows()
    vcap.release()


def main(camera_id):
    vcap = cv2.VideoCapture(camera_id)
    track_video(vcap)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_id", type=int, help="camera id")
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**parse_args())
