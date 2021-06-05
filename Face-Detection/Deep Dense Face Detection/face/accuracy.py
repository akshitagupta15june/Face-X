"""
Script for checking face detection algorithms accuracy
"""

import os

import shapely.geometry
import cv2
import numpy as np
import tqdm

import face.config
import face.utilities
import face.geometry
import face.models
import face.detection


def does_opencv_detect_face_correctly(image, face_bounding_box, cascade_classifier):

    detections = cascade_classifier.detectMultiScale(
        image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # There should be exactly one face detection in image
    if len(detections) != 1:

        return False

    else:

        left, top, width, height = detections[0]
        detection_bounding_box = shapely.geometry.box(left, top, left + width, top + height)

        is_detection_correct = face.geometry.get_intersection_over_union(
            face_bounding_box, detection_bounding_box) > 0.5

        return is_detection_correct


def check_opencv_accuracy(image_paths, bounding_boxes_map):

    detection_scores = []

    filters_path = os.path.expanduser("~/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    cascade_classifier = cv2.CascadeClassifier(filters_path)

    for path in tqdm.tqdm(image_paths):

        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

        image_bounding_box = shapely.geometry.box(0, 0, image.shape[1], image.shape[0])
        face_bounding_box = bounding_boxes_map[os.path.basename(path)]

        # Only try to search for faces if they are larger than 1% of image. If they are smaller,
        # ground truth bounding box is probably incorrect
        if face.geometry.get_intersection_over_union(image_bounding_box, face_bounding_box) > 0.01:

            value = 1 if does_opencv_detect_face_correctly(image, face_bounding_box, cascade_classifier) else 0
            detection_scores.append(value)

    print("OpenCV accuracy is {}".format(np.mean(detection_scores)))


def does_model_detect_face_correctly(image, face_bounding_box, model, configuration):

    detections = face.detection.FaceDetector(image, model, configuration).get_faces_detections()

    # There should be exactly one face detection in image
    if len(detections) != 1:

        return False

    else:

        is_detection_correct = face.geometry.get_intersection_over_union(
            face_bounding_box, detections[0].bounding_box) > 0.5

        return is_detection_correct


def check_model_accuracy(image_paths, bounding_boxes_map, file_path=None):

    detection_scores = []

    model = face.models.get_pretrained_vgg_model(face.config.image_shape)
    model.load_weights(face.config.model_path)

    for path in tqdm.tqdm(image_paths):

        image = face.utilities.get_image(path)

        image_bounding_box = shapely.geometry.box(0, 0, image.shape[1], image.shape[0])
        face_bounding_box = bounding_boxes_map[os.path.basename(path)]

        # Only try to search for faces if they are larger than 1% of image. If they are smaller,
        # ground truth bounding box probably is incorrect
        if face.geometry.get_intersection_over_union(image_bounding_box, face_bounding_box) > 0.01:

            value = 1 if does_model_detect_face_correctly(
                image, face_bounding_box, model, face.config.face_search_config) else 0

            detection_scores.append(value)

            if file_path is not None:

                with open(file_path, mode="a") as file:

                    file.write("{}\n".format(np.mean(detection_scores)))

    print("Model accuracy is {}".format(np.mean(detection_scores)))


def main():

    # dataset = "large_dataset"
    # dataset = "medium_dataset"
    dataset = "small_dataset"

    data_directory = os.path.join(face.config.data_directory, dataset)

    image_paths_file = os.path.join(data_directory, "training_image_paths.txt")
    bounding_boxes_file = os.path.join(data_directory, "training_bounding_boxes_list.txt")

    image_paths = [path.strip() for path in face.utilities.get_file_lines(image_paths_file)]
    bounding_boxes_map = face.geometry.get_bounding_boxes_map(bounding_boxes_file)

    # check_opencv_accuracy(image_paths, bounding_boxes_map)
    check_model_accuracy(image_paths, bounding_boxes_map, file_path="/tmp/face_accuracy_log.txt")


if __name__ == "__main__":

    main()
