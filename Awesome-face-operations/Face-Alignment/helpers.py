from collections import OrderedDict
import numpy as np

def rect_to__bb(rect):
    """
    Takes a bounding predicted by dlib and convert it to
    the format (x, y, w, h) as we would do normally on
    OpenCV
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    # return the list of (x, y)-coordinates
    return coords