import argparse
import cv2
import json
import logging
import logging.config
import numpy as np
import os

from utils import Parser, fetch_faces
from utils import metrics


# setup logger
config_file = os.path.abspath("../log/logging.conf")
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__file__)

# Drawn images will be saved in this directory
DST_DIR = "results"
if not os.path.exists(DST_DIR):
    logger.info(f"Creating {DST_DIR} directory")
    os.mkdir(DST_DIR)


def save_image(image, filename, groundtruth, preds):
    canvas = image.copy()
    for (x1, y1, x2, y2) in groundtruth:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x1, y1, x2, y2) in preds:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # save image
    dst_path = os.path.join(DST_DIR, filename)
    logger.info(f"saving image in {dst_path}")
    cv2.imwrite(dst_path, canvas)


def test_single_image(parser, threshold, save_random=True):
    # obtain groundtruth data
    metadata = parser.fetch_metadata()
    # filter to get people who only wear mask
    idxs = np.where(metadata["placement"] != "none")[0]
    metadata["bboxes"] = metadata["bboxes"][idxs]

    # do inference on mtcnn
    image = cv2.imread(parser.image_path)
    bboxes, _ = fetch_faces(image, return_landmarks=False)
    recall = metrics.recall(metadata["bboxes"], bboxes, threshold=threshold)

    # save results for ablation study
    if save_random and np.random.uniform() <= 0.10:
        filename = os.path.basename(parser.image_path)
        save_image(image, filename, metadata["bboxes"], bboxes)

    # recall for single inference image
    return recall


def test_mtcnn(base_dir, thresholds):
    # initialize data
    results = dict(zip(thresholds, [[] for _ in thresholds]))
    filenames = os.listdir(base_dir)
    logger.debug(f"Going to process {len(filenames)} images")

    # calculate individual recalls
    for i, filename in enumerate(filenames):
        try:
            parser = Parser(base_dir, filename)
        except FileNotFoundError:
            continue

        for th in thresholds:
            recall = test_single_image(parser, th, save_random=False)
            if recall is None:
                break
            results[th].append(recall)

        # log every n images
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1} images")

    # Notify reults
    for th, recall in results.items():
        logger.info(f"Mean recall (@IoU={th:0.0%}) = {np.mean(recall):0.3f}")

    return results


def main(base_dir):
    thresholds = np.linspace(0.1, 1, 10, endpoint=True)
    results = test_mtcnn(base_dir, thresholds)

    # get only mean recall
    for th, data in results.items():
        results[th] = np.mean(data)

    # save data
    with open("recall.json", "w") as outfile:
        logger.info("Saved results in recall.json")
        json.dump(results, outfile)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="path/to/labels")
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**parse_args())
