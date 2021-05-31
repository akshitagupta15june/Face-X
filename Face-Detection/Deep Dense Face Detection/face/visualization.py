"""
Script with a bunch useful visualization functions like showing output of data generators, crops predictions,
 heatmaps, etc. Useful both for debugging and quick demos.
"""

import os
import random

import vlogging
import numpy as np
import cv2
import tqdm

import face.utilities
import face.data_generators
import face.processing
import face.models
import face.config
import face.detection
import face.geometry


def log_data_batches(data_generator, logger):

    for _ in range(8):

        images, labels = next(data_generator)

        images = [image * 255 for image in images]
        images = [face.processing.scale_image_keeping_aspect_ratio(image, 100) for image in images]
        logger.info(vlogging.VisualRecord("Images batch", images, str(labels)))


def log_crops_predictions(data_generator, logger):

    model = face.models.get_pretrained_vgg_model(face.config.image_shape)
    model.load_weights(face.config.model_path)

    for _ in range(8):

        images, _ = next(data_generator)
        predictions = model.predict(images)

        images = [image * 255 for image in images]
        images = [face.processing.scale_image_keeping_aspect_ratio(image, 100) for image in images]

        logger.info(vlogging.VisualRecord("Crops predictions", images, str(predictions)))


def log_heatmaps(image_paths_file, logger):

    model = face.models.get_pretrained_vgg_model(face.config.image_shape)
    model.load_weights(face.config.model_path)

    paths = [path.strip() for path in face.utilities.get_file_lines(image_paths_file)]
    random.shuffle(paths)

    for path in tqdm.tqdm(paths[:10]):

        image = face.utilities.get_image(path)

        heatmap = face.detection.HeatmapComputer(
            image, model, face.config.face_search_config).get_heatmap()

        scaled_images = [255 * image, 255 * heatmap]
        scaled_images = [face.processing.scale_image_keeping_aspect_ratio(image, 200) for image in scaled_images]

        logger.info(vlogging.VisualRecord("Heatmap", scaled_images, str(image.shape)))


def log_face_detections(image_paths_file, logger):

    model = face.models.get_pretrained_vgg_model(face.config.image_shape)
    model.load_weights(face.config.model_path)

    paths = [path.strip() for path in face.utilities.get_file_lines(image_paths_file)]
    random.shuffle(paths)

    for path in tqdm.tqdm(paths[:10]):

        image = face.utilities.get_image(path)

        detections = face.detection.FaceDetector(
            image, model, face.config.face_search_config).get_faces_detections()

        for face_detection in detections:

            face.geometry.draw_bounding_box(image, face_detection.bounding_box, color=(0, 1, 0), thickness=4)

        logger.info(vlogging.VisualRecord("Detections", image * 255,
                                          "{} - {}".format(path, str(image.shape))))


def main():

    logger = face.utilities.get_logger(face.config.log_path)

    # dataset = "large_dataset"
    # dataset = "medium_dataset"
    dataset = "small_dataset"

    data_directory = os.path.join(face.config.data_directory, dataset)

    image_paths_file = os.path.join(data_directory, "training_image_paths.txt")
    bounding_boxes_file = os.path.join(data_directory, "training_bounding_boxes_list.txt")

    generator = face.data_generators.get_batches_generator(
        image_paths_file, bounding_boxes_file, batch_size=8, crop_size=face.config.crop_size)

    # log_data_batches(generator, logger)
    # log_crops_predictions(generator, logger)
    # log_heatmaps(image_paths_file, logger)
    log_face_detections(image_paths_file, logger)


if __name__ == "__main__":

    main()
