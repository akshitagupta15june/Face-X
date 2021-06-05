"""
Tests for face.processing module
"""

import numpy as np

import face.processing


def test_scale_image_keeping_aspect_ratio_vertial_image():

    image = np.zeros(shape=[10, 20])
    target_size = 30

    # Vertical dimension smaller
    rescaled_image = face.processing.scale_image_keeping_aspect_ratio(image, target_size)

    assert (30, 60) == rescaled_image.shape


def test_scale_image_keeping_aspect_ratio_horizontal_image():

    image = np.zeros(shape=[10, 5])
    target_size = 20

    # Vertical dimension smaller
    rescaled_image = face.processing.scale_image_keeping_aspect_ratio(image, target_size)

    assert (40, 20) == rescaled_image.shape


def test_get_scaled_image_square_image():

    image = np.zeros(shape=[10, 10])
    scale = 2

    scaled_image = face.processing.get_scaled_image(image, scale)

    assert (20, 20) == scaled_image.shape


def test_get_scaled_image_horizontal_image():

    image = np.zeros(shape=[10, 30])
    scale = 0.3

    scaled_image = face.processing.get_scaled_image(image, scale)

    assert (3, 9) == scaled_image.shape


def test_get_scaled_image_vertical_image():

    image = np.zeros(shape=[40, 20])
    scale = 0.4

    scaled_image = face.processing.get_scaled_image(image, scale)

    assert (16, 8) == scaled_image.shape


def test_get_smallest_expected_face_size_min_size_is_the_cap():

    image_shape = [100, 200]
    min_face_size = 50
    min_face_to_image_ratio = 0.1

    expected = 50
    actual = face.processing.get_smallest_expected_face_size(image_shape, min_face_size, min_face_to_image_ratio)

    assert expected == actual


def test_get_smallest_expected_face_size_horizontal_image_size_is_the_cap():

    image_shape = [100, 50]
    min_face_size = 1
    min_face_to_image_ratio = 0.1

    expected = 5
    actual = face.processing.get_smallest_expected_face_size(image_shape, min_face_size, min_face_to_image_ratio)

    assert expected == actual


def test_get_smallest_expected_face_size_vertical_image_size_is_the_cap():

    image_shape = [200, 500]
    min_face_size = 1
    min_face_to_image_ratio = 0.1

    expected = 20
    actual = face.processing.get_smallest_expected_face_size(image_shape, min_face_size, min_face_to_image_ratio)

    assert expected == actual


def test_get_smallest_expected_face_size_3D_image_shape():

    image_shape = [200, 500, 3]
    min_face_size = 1
    min_face_to_image_ratio = 0.1

    expected = 20
    actual = face.processing.get_smallest_expected_face_size(image_shape, min_face_size, min_face_to_image_ratio)

    assert expected == actual