import unittest
import numpy as np
from image_to_numpy import load_image_file
import matplotlib.pyplot as plt
import image_to_numpy

class TestLoadImageFile(unittest.TestCase):

    # def test_jpeg_rotation(self):
    #     # Make sure all Portrait test assets are auto-rotated correctly
    #     for i in range(9):
    #         img_jpg = load_image_file(f"Portrait_{i}.jpg") 
    #         ref_img = np.load(f"Portrait_{i}.jpg.npy")
    #         self.assertTrue(np.array_equal(ref_img, img_jpg))

    #     # Make sure all Landscape test assets are auto-rotated correctly
    #     for i in range(9):
    #         img_jpg = load_image_file(f"Landscape_{i}.jpg")
    #         ref_img = np.load(f"Landscape_{i}.jpg.npy")
    #         self.assertTrue(np.array_equal(ref_img, img_jpg))

    def test_jpeg_no_exif(self):
        # Can we load a jpeg with no metadata without crashing?
        img_jpg = load_image_file("Portrait_no_exif.jpg")
        self.assertEqual(img_jpg.shape, (1200, 1800, 3))

    def test_png(self):
        # Can we load a non-jpeg file with no metadata?
        img_png = load_image_file("Portrait_8.png")
        self.assertEqual(img_png.shape, (1800, 1200, 3))
