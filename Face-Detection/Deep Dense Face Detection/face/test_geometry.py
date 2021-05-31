"""
Tests for face.geometry module
"""

import mock

import shapely.geometry

import face.geometry


def test_get_bounding_box():

    left = 10
    upper = 20
    width = 5
    height = 10

    expected = shapely.geometry.box(10, 20, 15, 30)
    actual = face.geometry.get_bounding_box(left, upper, width, height)

    assert expected.equals(actual)


def test_get_bounding_boxes_map():

    file_lines = "202599\nimage_id x_1 y_1 width height\n000001.jpg    95  71 226 313\n000002.jpg    72  94 221 306\n"
    mock_opener = mock.mock_open(read_data=file_lines)

    kwargs = {"open": mock_opener}

    first_bounding_box = shapely.geometry.box(95, 71, 95 + 226, 71 + 313)
    second_bounding_box = shapely.geometry.box(72, 94, 72 + 221, 94 + 306)

    expected = {
        "000001.jpg": first_bounding_box,
        "000002.jpg": second_bounding_box
    }

    actual = face.geometry.get_bounding_boxes_map("whatever", **kwargs)

    assert "000001.jpg" in expected
    assert "000002.jpg" in expected

    assert first_bounding_box.equals(actual["000001.jpg"])
    assert second_bounding_box.equals(actual["000002.jpg"])


def test_get_intersection_over_union_simple_intersection():

    first_polygon = shapely.geometry.box(10, 10, 20, 20)
    second_polygon = shapely.geometry.box(10, 10, 15, 15)

    assert 0.25 == face.geometry.get_intersection_over_union(first_polygon, second_polygon)
    assert 0.25 == face.geometry.get_intersection_over_union(second_polygon, first_polygon)


def test_get_intersection_over_union_non_intersecting_polygons():

    first_polygon = shapely.geometry.box(10, 10, 20, 20)
    second_polygon = shapely.geometry.box(100, 100, 150, 150)

    assert 0 == face.geometry.get_intersection_over_union(first_polygon, second_polygon)
    assert 0 == face.geometry.get_intersection_over_union(second_polygon, first_polygon)


def test_get_scale_horizontal_box():

    box = shapely.geometry.box(10, 20, 50, 30)
    target_size = 5

    assert 0.5 == face.geometry.get_scale(box, target_size)


def test_get_scale_vertical_box():

    box = shapely.geometry.box(10, 20, 50, 120)
    target_size = 10

    assert 0.25 == face.geometry.get_scale(box, target_size)


def test_get_scaled_bounding_box_long_box():

    box = shapely.geometry.box(10, 20, 50, 30)
    scale = 3

    expected = shapely.geometry.box(30, 60, 150, 90)
    actual = face.geometry.get_scaled_bounding_box(box, scale)

    assert expected.equals(actual)


def test_get_scaled_bounding_box_tall_box():

    box = shapely.geometry.box(10, 20, 50, 100)
    scale = 0.5

    expected = shapely.geometry.box(5, 10, 25, 50)
    actual = face.geometry.get_scaled_bounding_box(box, scale)

    assert expected.equals(actual)


def test_flip_bounding_box_about_vertical_axis_box_to_left_of_axis():

    box = shapely.geometry.box(10, 20, 60, 40)
    image_shape = [200, 100]

    expected = shapely.geometry.box(40, 20, 90, 40)
    actual = face.geometry.flip_bounding_box_about_vertical_axis(box, image_shape)

    assert expected.equals(actual)


def test_flip_bounding_box_about_vertical_axis_box_to_right_of_axis():

    box = shapely.geometry.box(30, 10, 70, 50)
    image_shape = [200, 80]

    expected = shapely.geometry.box(10, 10, 50, 50)
    actual = face.geometry.flip_bounding_box_about_vertical_axis(box, image_shape)

    assert expected.equals(actual)


def test_flip_bounding_box_box_centered_on_axis():

    box = shapely.geometry.box(30, 10, 90, 90)
    image_shape = [200, 120]

    expected = box
    actual = face.geometry.flip_bounding_box_about_vertical_axis(box, image_shape)

    assert expected.equals(actual)