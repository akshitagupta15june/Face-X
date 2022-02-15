"""
Code for working with Celeb+ dataset
"""

import os
import shutil
import subprocess
import glob

import face.download
import face.utilities
import face.geometry


class DatasetBuilder:
    """
    Class for downloading Celeb+ data and preparing datasets from it.
    """

    def __init__(self, data_directory):

        self.data_directory = data_directory
        self.bounding_boxes_path = os.path.join(self.data_directory, "all_bounding_boxes.txt")

    def build_datasets(self):

        shutil.rmtree(self.data_directory, ignore_errors=True)
        os.makedirs(self.data_directory, exist_ok=True)

        self._get_images()
        self._get_bounding_boxes()

        image_paths = self._get_image_paths(self.data_directory)
        bounding_boxes_map = self._get_bounding_boxes_map(self.bounding_boxes_path)

        datasets_dirs = ["large_dataset", "medium_dataset", "small_dataset"]

        large_dataset_split = [0, 180000, 190000, len(image_paths)]
        medium_dataset_split = [0, 10000, 20000, 30000]
        small_dataset_split = [0, 1000, 2000, 3000]

        splits = [large_dataset_split, medium_dataset_split, small_dataset_split]

        for dataset_dir, splits in zip(datasets_dirs, splits):

            directory = os.path.join(self.data_directory, dataset_dir)
            DataSubsetBuilder(directory, image_paths, bounding_boxes_map, splits).build()

    def _get_images(self):

        image_archives_urls = [
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABQwEE5YX5jTFGXjo0f9glIa/Img/img_celeba.7z/img_celeba.7z.001?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADxKopMA7g_Ka2o7X7B8jiHa/Img/img_celeba.7z/img_celeba.7z.002?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABSqeGALxGo1sXZ-ZizRFa5a/Img/img_celeba.7z/img_celeba.7z.003?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADBal8W3N9AYwYuqwTtA_fQa/Img/img_celeba.7z/img_celeba.7z.004?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACJaDb7rWNFcCKqcFjFjUlHa/Img/img_celeba.7z/img_celeba.7z.005?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACcD0ZMO36zVaIfLGLKtrq4a/Img/img_celeba.7z/img_celeba.7z.006?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAhuX-S5ULmy8GII6jlZFb9a/Img/img_celeba.7z/img_celeba.7z.007?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAUtign0NJIV8fRK7xt6TIEa/Img/img_celeba.7z/img_celeba.7z.008?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACJsmneLOU5xMB2qmnJA0AGa/Img/img_celeba.7z/img_celeba.7z.009?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAfZVSjBlkPr5e5GYMek50_a/Img/img_celeba.7z/img_celeba.7z.010?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA6-edxuJyMBoGZqTdl28bpa/Img/img_celeba.7z/img_celeba.7z.011?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABMLOgnvv8DKxt4UvULSAoha/Img/img_celeba.7z/img_celeba.7z.012?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABOeeqqAzZEY6jDwTdOUTqRa/Img/img_celeba.7z/img_celeba.7z.013?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADuEM2h2qG_L0UbUTViRH5Da/Img/img_celeba.7z/img_celeba.7z.014?dl=1"
        ]

        filenames = [os.path.basename(url).split("?")[0] for url in image_archives_urls]
        paths = [os.path.join(self.data_directory, filename) for filename in filenames]

        # Download image archives
        for url, path in zip(image_archives_urls, paths):

            face.download.Downloader(url, path).download()

        # Extract assets
        subprocess.call(["7z", "x", paths[0], "-o" + self.data_directory])

        # Delete image archives
        for path in paths:

            os.remove(path)

    def _get_bounding_boxes(self):

        url = "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACL5lLyHMAHvFA8W17JDahma/Anno/list_bbox_celeba.txt?dl=1"
        face.download.Downloader(url, self.bounding_boxes_path).download()

    def _get_image_paths(self, data_directory):

        image_paths = glob.glob(os.path.join(data_directory, "**/*.jpg"), recursive=True)
        image_paths = [os.path.abspath(path) for path in image_paths]
        return image_paths

    def _get_bounding_boxes_map(self, bounding_boxes_path):

        bounding_boxes_lines = face.utilities.get_file_lines(bounding_boxes_path)[2:]
        bounding_boxes_map = {}

        for line in bounding_boxes_lines:

            tokens = line.split()

            filename = tokens[0]

            integer_tokens = [round(token) for token in tokens[1:]]
            bounding_box = face.geometry.get_bounding_box(*integer_tokens)

            bounding_boxes_map[filename] = bounding_box

        return bounding_boxes_map


class DataSubsetBuilder:
    """
    A helper class for DatasetBuilder
    """

    def __init__(self, directory, image_paths, bounding_boxes_map, splits):

        self.data_directory = directory
        self.image_paths = image_paths
        self.bounding_boxes_map = bounding_boxes_map
        self.splits = splits

    def build(self):

        shutil.rmtree(self.data_directory, ignore_errors=True)
        os.makedirs(self.data_directory, exist_ok=True)

        training_image_paths = self.image_paths[self.splits[0]:self.splits[1]]
        validation_image_paths = self.image_paths[self.splits[1]:self.splits[2]]
        test_image_paths = self.image_paths[self.splits[2]:self.splits[3]]

        splitted_image_paths = [training_image_paths, validation_image_paths, test_image_paths]

        prefixes = ["training_", "validation_", "test_"]

        images_list_file_names = [prefix + "image_paths.txt" for prefix in prefixes]
        images_list_file_paths = [os.path.join(self.data_directory, filename)
                                  for filename in images_list_file_names]

        # Create files with image paths
        for image_list_path, image_paths in zip(images_list_file_paths, splitted_image_paths):
            self._create_paths_file(image_list_path, image_paths)

        bounding_boxes_list_file_names = [prefix + "bounding_boxes_list.txt" for prefix in prefixes]
        bounding_boxes_list_file_paths = [os.path.join(self.data_directory, filename)
                                          for filename in bounding_boxes_list_file_names]

        # Create files with bounding boxes lists
        for bounding_box_list_path, image_paths in zip(bounding_boxes_list_file_paths, splitted_image_paths):
            self._create_bounding_boxes_file(bounding_box_list_path, image_paths, self.bounding_boxes_map)

    def _create_paths_file(self, file_path, image_paths):

        paths = [path + "\n" for path in image_paths]

        with open(file_path, "w") as file:

            file.writelines(paths)

    def _create_bounding_boxes_file(self, file_path, image_paths, bounding_boxes_map):

        image_paths = [os.path.basename(path) for path in image_paths]

        header = str(len(image_paths)) + "\nimage_id x_1 y_1 width height\n"

        with open(file_path, "w") as file:

            file.write(header)

            for image_path in image_paths:

                bounds = [round(value) for value in bounding_boxes_map[image_path].bounds]

                x = bounds[0]
                y = bounds[1]
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]

                line = "{}\t{} {} {} {}\n".format(image_path, x, y, width, height)
                file.write(line)
