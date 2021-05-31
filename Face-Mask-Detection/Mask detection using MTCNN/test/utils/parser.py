import os
import logging
import numpy as np

from xml.etree import ElementTree as ET


# setup logger
parent_dir, filename = os.path.split(__file__)
base_dir = os.path.basename(parent_dir)
logger = logging.getLogger(os.path.join(base_dir, filename))


class Parser(object):

    def __init__(self, base_dir, xml_file):
        self.base_dir = base_dir
        xml_path = os.path.join(base_dir, xml_file)
        self.root = ET.parse(xml_path).getroot()
        self.get_image_path()

        # some files in the dataset do not exist
        if not os.path.exists(self.image_path):
            logger.error(f"Could not find {self.image_path}")
            raise FileNotFoundError(f"{self.image_path} does not exist")

    @staticmethod
    def _fetch_bounding_box(obj):
        attrib = obj.find("bndbox")
        xmin = int(attrib.find("xmin").text)
        ymin = int(attrib.find("ymin").text)
        xmax = int(attrib.find("xmax").text)
        ymax = int(attrib.find("ymax").text)
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def _fetch_difficulty(obj):
        return int(obj.find("difficult").text)

    @staticmethod
    def _fetch_placement(obj):
        return obj.find("name").text

    def get_image_path(self):
        image_dir = self.root.find("folder").text
        filename = self.root.find("filename").text
        parent_dir = os.path.dirname(os.path.dirname(self.base_dir))
        self.image_path = os.path.join(parent_dir, image_dir, filename)

    def fetch_metadata(self):
        # initialize data
        bboxes = []
        difficulty = []
        placement = []
        # metadata is located withing each object attribute
        for obj in self.root.findall("object"):
            bboxes.append(self._fetch_bounding_box(obj))
            difficulty.append(self._fetch_difficulty(obj))
            placement.append(self._fetch_placement(obj))

        # build data structure
        metadata = {
            "bboxes": np.array(bboxes).astype("int"), "difficult": np.array(difficulty), "placement": np.array(placement)
        }
        return metadata
