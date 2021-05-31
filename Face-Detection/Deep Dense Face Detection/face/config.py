"""
Config file with common constants
"""

# Path to data directory
data_directory = "../../data/faces/"

# Log file path
log_path = "/tmp/faces/log.html"

# Batch size to be used by prediction models
batch_size = 64

# Size of crops data generators should return
crop_size = 64

# Size of inputs models are trained on
image_shape = (crop_size, crop_size, 3)

# Stride to be used to sample crops from images
stride = 8


class SingleScaleFaceSearchConfiguration:
    """
    A simple class that bundles together common face search parameters for single scale searches.
    """

    def __init__(self, crop_size, stride, batch_size):
        """
        Constructor
        :param crop_size: size of crops used to search for faces
        :param stride: stride between successive crops
        :param batch_size: batch size used by predictive model
        """

        self.crop_size = crop_size
        self.stride = stride
        self.batch_size = batch_size


# Default face search configuration
single_scale_face_search_config = SingleScaleFaceSearchConfiguration(
    crop_size=crop_size, stride=stride, batch_size=batch_size)


# Minimum size of a face, in pixels, we want to search for
min_face_size = 50

# Minimum ratio of face to image we want to search for. For image of size x (along smaller dimension)
# we only want to consider regions not smaller than min_face_to_image_ratio times x as possible face candidates
min_face_to_image_ratio = 0.1

# Ratio by which image should be scaled down on each successive move on image pyramid
image_rescaling_ratio = 0.8


class FaceSearchConfiguration(SingleScaleFaceSearchConfiguration):
    """
    A simple class that bundles together common multi scale face search parameters
    """

    def __init__(self, crop_size, stride, batch_size, min_face_size, min_face_to_image_ratio, image_rescaling_ratio):
        """
        Constructor
        :param crop_size: size of crops used to search for faces
        :param stride: stride between successive crops
        :param batch_size: batch size used by predictive model
        :param min_face_size: minimum size of a face, in pixels, we want to search for
        :param min_face_to_image_ratio: minimum ratio of face to image we want to search for.
        For image of size x (along smaller dimension) we only want to consider regions not smaller than
        min_face_to_image_ratio times x as possible face candidates.
        In practice the larger value of min_face_size and (min_face_to_image_ratio x smallest image dimension) is used
         as smallest region used to search for faces.
        :param image_rescaling_ratio: ratio by which image should be scaled down on each
        successive move on image pyramid
        """

        super().__init__(crop_size, stride, batch_size)

        self.min_face_size = min_face_size
        self.min_face_to_image_ratio = min_face_to_image_ratio
        self.image_rescaling_ratio = image_rescaling_ratio


# Default multi scale face search configuration
face_search_config = FaceSearchConfiguration(
    crop_size=crop_size, stride=stride, batch_size=batch_size,
    min_face_size=min_face_size, min_face_to_image_ratio=min_face_to_image_ratio,
    image_rescaling_ratio=image_rescaling_ratio)

# Path to model file
model_path = "../../data/faces/models/model.h5"
