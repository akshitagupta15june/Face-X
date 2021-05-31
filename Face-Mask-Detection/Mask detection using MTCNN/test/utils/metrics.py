import os
import logging


# setup logger
parent_dir, filename = os.path.split(__file__)
base_dir = os.path.basename(parent_dir)
logger = logging.getLogger(os.path.join(base_dir, filename))


def _calculate_true_positives(groundtruth, preds, threshold):
    tp = 0
    for bbox_gt in groundtruth:
        for bbox_pred in preds:
            if intersection_over_union(bbox_gt, bbox_pred) >= threshold:
                tp += 1
    return tp


def recall(groundtruth, preds, threshold=0.5):
    """
    The recall of a model is given by the ratio of true object detections
    to the total number of objects in the dataset
    """
    true_positives = _calculate_true_positives(groundtruth, preds, threshold)
    # We can ignore False Negatives as TP + FN is the same as groundtruth
    # detections, For the moment, we won-t take those into account
    total_detections = len(groundtruth)
    if total_detections == 0:
        logger.info("Groundtruth image did not have any detections")
        return None
    return true_positives / total_detections


def intersection_over_union(bbox_a, bbox_b):
    # obtain coords of intersection rectangle
    xa = max(bbox_a[0], bbox_b[0])
    ya = max(bbox_a[1], bbox_b[1])
    xb = min(bbox_a[2], bbox_b[2])
    yb = min(bbox_a[3], bbox_b[3])

    # compute area of intersection rectangle
    inter_area = abs(max(xb - xa, 0) * max(yb - ya, 0))
    if inter_area == 0:
        return 0

    # compute area of groundtruth and predictions rectangles
    area_a = abs((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    area_b = abs((bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))

    # compute intersection over union
    iou = inter_area / float(area_a + area_b - inter_area)
    return iou
