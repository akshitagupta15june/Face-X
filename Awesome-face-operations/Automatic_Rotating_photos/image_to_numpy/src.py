import PIL.Image
import PIL.ImageOps
import numpy as np


def exif_transpose(img):
    """
    If an image has an Exif Orientation tag, transpose the image
    accordingly.

    Note: Very recent versions of Pillow have an internal version
    of this function. So this is only needed if Pillow isn't at the
    latest version.

    :param image: The image to transpose.
    :return: An image.
    """
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    Defaults to returning the image data as a 3-channel array of 8-bit data. That is
    controlled by the mode parameter.

    Supported modes:
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, black and white)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        CMYK (4x8-bit pixels, color separation)
        YCbCr (3x8-bit pixels, color video format)
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)

    :param file: image file name or file object to load
    :param mode: format to convert the image to - 'RGB' (8-bit RGB, 3 channels), 'L' (black and white)
    :return: image contents as numpy array
    """

    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return np.array(img)
