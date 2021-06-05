"""
Tests for face.detection module
"""

import mock

import numpy as np
import shapely.geometry
import pytest

import face.detection
import face.config
import face.geometry


def test_get_face_candidates_generator_raises_on_stride_larger_than_crop_size():

    with pytest.raises(ValueError):

        generator = face.detection.get_face_candidates_generator(
            np.zeros(shape=[10, 10]), crop_size=4, stride=5, batch_size=4)

        next(generator)


def test_get_face_candidates_generator_returns_no_batches_when_image_smaller_than_crops():

    generator = face.detection.get_face_candidates_generator(
        np.zeros(shape=[2, 2]), crop_size=4, stride=4, batch_size=4)

    with pytest.raises(StopIteration):

        next(generator)


def test_get_face_candidates_generator_candidates_single_row_crops():

    image = np.arange(40).reshape([4, 10])
    crop_size = 4
    stride = 3

    generator = face.detection.get_face_candidates_generator(image, crop_size, stride, batch_size=2)

    # Get first batch
    batch = next(generator)

    assert 2 == len(batch)

    # Assert properties of first candidate
    assert 0 == np.min(batch[0].cropped_image)
    assert 33 == np.max(batch[0].cropped_image)

    assert shapely.geometry.box(0, 0, 4, 4) == batch[0].crop_coordinates
    assert shapely.geometry.box(0, 0, 4, 4) == batch[0].focus_coordinates

    # Assert properties of second candidate
    assert 3 == np.min(batch[1].cropped_image)
    assert 36 == np.max(batch[1].cropped_image)

    assert shapely.geometry.box(3, 0, 7, 4) == batch[1].crop_coordinates
    assert shapely.geometry.box(3, 0, 7, 4) == batch[1].focus_coordinates

    # Get second batch
    batch = next(generator)

    assert 1 == len(batch)

    # Assert properties of third candidate
    assert 6 == np.min(batch[0].cropped_image)
    assert 39 == np.max(batch[0].cropped_image)

    assert shapely.geometry.box(6, 0, 10, 4) == batch[0].crop_coordinates
    assert shapely.geometry.box(6, 0, 10, 4) == batch[0].focus_coordinates

    # There should be no more batches available
    with pytest.raises(StopIteration):

        next(generator)


def test_get_face_candidates_generator_single_column_crops():

    image = np.arange(75).reshape([15, 5])
    crop_size = 5
    stride = 4

    generator = face.detection.get_face_candidates_generator(image, crop_size, stride, batch_size=2)

    # Get first batch
    batch = next(generator)

    assert 2 == len(batch)

    # Assert properties of first candidate
    assert 0 == np.min(batch[0].cropped_image)
    assert 24 == np.max(batch[0].cropped_image)

    assert shapely.geometry.box(0, 0, 5, 5) == batch[0].crop_coordinates
    assert shapely.geometry.box(0, 0, 5, 5) == batch[0].focus_coordinates

    # Assert properties of second candidate
    assert 20 == np.min(batch[1].cropped_image)
    assert 44 == np.max(batch[1].cropped_image)

    assert shapely.geometry.box(0, 4, 5, 9) == batch[1].crop_coordinates
    assert shapely.geometry.box(0, 4, 5, 9) == batch[1].focus_coordinates

    # Get second batch
    batch = next(generator)

    assert 1 == len(batch)

    # Assert properties of third candidate
    assert 40 == np.min(batch[0].cropped_image)
    assert 64 == np.max(batch[0].cropped_image)

    assert shapely.geometry.box(0, 8, 5, 13) == batch[0].crop_coordinates
    assert shapely.geometry.box(0, 8, 5, 13) == batch[0].focus_coordinates

    # There should be no more batches available
    with pytest.raises(StopIteration):
        next(generator)


def test_get_face_candidates_generator_simple_grid():

    image = np.arange(100).reshape([10, 10])
    crop_size = 5
    stride = 4

    generator = face.detection.get_face_candidates_generator(image, crop_size, stride, batch_size=3)

    # Get first batch
    batch = next(generator)

    assert 3 == len(batch)

    # Assert properties of first candidate
    assert 0 == np.min(batch[0].cropped_image)
    assert 44 == np.max(batch[0].cropped_image)

    assert shapely.geometry.box(0, 0, 5, 5) == batch[0].crop_coordinates
    assert shapely.geometry.box(0, 0, 5, 5) == batch[0].focus_coordinates

    # Assert properties of second candidate
    assert 4 == np.min(batch[1].cropped_image)
    assert 48 == np.max(batch[1].cropped_image)

    assert shapely.geometry.box(4, 0, 9, 5) == batch[1].crop_coordinates
    assert shapely.geometry.box(4, 0, 9, 5) == batch[1].focus_coordinates

    # Assert properties of third candidate
    assert 40 == np.min(batch[2].cropped_image)
    assert 84 == np.max(batch[2].cropped_image)

    assert shapely.geometry.box(0, 4, 5, 9) == batch[2].crop_coordinates
    assert shapely.geometry.box(0, 4, 5, 9) == batch[2].focus_coordinates

    # Get second batch
    batch = next(generator)

    assert 1 == len(batch)

    # Assert properties of fourth candidate
    assert 44 == np.min(batch[0].cropped_image)
    assert 88 == np.max(batch[0].cropped_image)

    assert shapely.geometry.box(4, 4, 9, 9) == batch[0].crop_coordinates
    assert shapely.geometry.box(4, 4, 9, 9) == batch[0].focus_coordinates

    # There should be no more batches available
    with pytest.raises(StopIteration):
        next(generator)


def test_get_heatmap_single_batch():

    image = np.zeros(shape=[10, 10])

    mock_model = mock.Mock()
    mock_model.predict.return_value = [0.2, 0.4, 0.6, 0.8]

    configuration = face.config.SingleScaleFaceSearchConfiguration(crop_size=5, stride=4, batch_size=4)

    expected_heatmap = np.zeros(shape=[10, 10])
    expected_heatmap[:4, :4] = 0.2
    expected_heatmap[:4, 4:9] = 0.4
    expected_heatmap[4:9, :4] = 0.6
    expected_heatmap[4:9, 4:9] = 0.8

    computer = face.detection.SingleScaleHeatmapComputer(image, mock_model, configuration)
    actual_heatmap = computer.get_heatmap()

    assert np.allclose(actual_heatmap, expected_heatmap)


class TestUniqueDetectionsComputer:

    def test_non_maximum_suppression_one_group_only(self):

        bounding_boxes = [
            shapely.geometry.box(0, 0, 10, 10),
            shapely.geometry.box(1, 1, 11, 11),
            shapely.geometry.box(2, 2, 12, 12)
        ]

        face_detections = [
            face.detection.FaceDetection(bounding_boxes[0], 0.9),
            face.detection.FaceDetection(bounding_boxes[1], 0.98),
            face.detection.FaceDetection(bounding_boxes[2], 0.95)
        ]

        iou_threshold = 0.5

        expected_results = [
            face.detection.FaceDetection(bounding_boxes[1], 0.98)
        ]

        actual_results = face.detection.UniqueDetectionsComputer.non_maximum_suppression(
            face_detections, iou_threshold)

        assert expected_results == actual_results

    def test_non_maximum_suppression_two_groups(self):

        bounding_boxes = [
            shapely.geometry.box(0, 0, 10, 10),
            shapely.geometry.box(1, 1, 11, 11),
            shapely.geometry.box(2, 2, 12, 12),
            shapely.geometry.box(100, 100, 110, 110),
            shapely.geometry.box(101, 101, 111, 111)
        ]

        face_detections = [
            face.detection.FaceDetection(bounding_boxes[0], 0.9),
            face.detection.FaceDetection(bounding_boxes[1], 0.98),
            face.detection.FaceDetection(bounding_boxes[2], 0.95),
            face.detection.FaceDetection(bounding_boxes[3], 0.9),
            face.detection.FaceDetection(bounding_boxes[4], 0.95)
        ]

        iou_threshold = 0.5

        expected_results = [
            face.detection.FaceDetection(bounding_boxes[1], 0.98),
            face.detection.FaceDetection(bounding_boxes[4], 0.95)
        ]

        actual_results = face.detection.UniqueDetectionsComputer.non_maximum_suppression(
            face_detections, iou_threshold)

        assert expected_results == actual_results

    def test_averaging_one_group_only(self):

        bounding_boxes = [
            shapely.geometry.box(0, 0, 10, 10),
            shapely.geometry.box(1, 1, 11, 11),
            shapely.geometry.box(2, 2, 12, 12)
        ]

        face_detections = [
            face.detection.FaceDetection(bounding_boxes[0], 0.9),
            face.detection.FaceDetection(bounding_boxes[1], 0.98),
            face.detection.FaceDetection(bounding_boxes[2], 0.95)
        ]

        iou_threshold = 0.5

        expected_results = [
            face.detection.FaceDetection(bounding_boxes[1], 0.98)
        ]

        actual_results = face.detection.UniqueDetectionsComputer.averaging(
            face_detections, iou_threshold)

        assert expected_results == actual_results

    def test_averaging_two_groups(self):

        bounding_boxes = [
            shapely.geometry.box(0, 0, 10, 10),
            shapely.geometry.box(1, 1, 11, 11),
            shapely.geometry.box(2, 2, 12, 12),
            shapely.geometry.box(100, 100, 110, 110),
            shapely.geometry.box(102, 98, 108, 106)
        ]

        face_detections = [
            face.detection.FaceDetection(bounding_boxes[0], 0.9),
            face.detection.FaceDetection(bounding_boxes[1], 0.98),
            face.detection.FaceDetection(bounding_boxes[2], 0.95),
            face.detection.FaceDetection(bounding_boxes[3], 0.9),
            face.detection.FaceDetection(bounding_boxes[4], 0.95)
        ]

        iou_threshold = 0.25

        expected_results = [
            face.detection.FaceDetection(bounding_boxes[1], 0.98),
            face.detection.FaceDetection(shapely.geometry.box(101, 99, 109, 108), 0.95)
        ]

        actual_results = face.detection.UniqueDetectionsComputer.averaging(
            face_detections, iou_threshold)

        assert expected_results == actual_results
