"""
This scripts downloads and preprocesses Celeb+ data used for training and testing
"""

import face.datasets.celeb
import face.config


def main():

    face.datasets.celeb.DatasetBuilder(face.config.data_directory).build_datasets()


if __name__ == "__main__":

    main()
