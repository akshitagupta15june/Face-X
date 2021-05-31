
from __future__ import print_function
import gif2numpy
import cv2
from matplotlib import pyplot as plt
import numpy as np

image = "glow_mini.gif"
frames, exts, image_specs = gif2numpy.convert(image)
sprites=np.array(frames)
np.save('glow_mini.npy', sprites)

'''
Install: pip install gif2numpy
Run: python3 giftonpy.py
'''
