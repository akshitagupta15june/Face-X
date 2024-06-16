import yaml
import os
import logging
from bing_image_downloader import downloader


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content


def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory is created at {dir_path}")


def data_download(name_of_image,limits):
    downloader.download(name_of_image, limit=limits,  output_dir='new_dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
    logging.info("Data has been downloaded.")


