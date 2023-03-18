import ctypes

from wand.api import library
from wand.color import Color
from wand.drawing import Drawing
from wand.image import Image

# Tell Python about C library
library.MagickPolaroidImage.argtypes = (ctypes.c_void_p,  # MagickWand *
                                        ctypes.c_void_p,  # DrawingWand *
                                        ctypes.c_double)  # Double

library.MagickSetImageBorderColor.argtypes = (ctypes.c_void_p,  # MagickWand *
                                              ctypes.c_void_p)  # PixelWand *


# Define FX method. See MagickPolaroidImage in wand/magick-image.c
def polaroid(wand, context, angle=0.0):
    if not isinstance(wand, Image):
        raise TypeError('wand must be instance of Image, not ' + repr(wand))
    if not isinstance(context, Drawing):
        raise TypeError('context must be instance of Drawing, not ' + repr(context))
    library.MagickPolaroidImage(wand.wand,
                                context.resource,
                                angle)
with Image(filename=r'.\R.jpeg') as image:
    # Assigne border color
    with Color('white') as white:
        library.MagickSetImageBorderColor(image.wand, white.resource)
    with Drawing() as annotation:
        # ... Optional caption text here ...
        polaroid(image, annotation)
    image.save(filename=r'.\your_image.jpeg')