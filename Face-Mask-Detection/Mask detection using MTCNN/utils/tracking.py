import os
import logging
import numpy as np

from collections import OrderedDict
from scipy.spatial import distance


# setup logger
parent_dir, filename = os.path.split(__file__)
base_dir = os.path.basename(parent_dir)
logger = logging.getLogger(os.path.join(base_dir, filename))


class CentroidTracker(object):

    def __init__(self, max_detections=30):
        self.next_id = 0
        # Objects will be a collection of Person class instances that we will be
        # analyzing
        self.objects = OrderedDict()
        # If a Person instance cannot be recovered in the limit defined by
        # max_detections, then delete it from the register
        self.disappeared = OrderedDict()
        self.max_detections = max_detections

    def _fetch_centroids(self):
        centroids = [person.centroid for person in self.objects.values()]
        return np.array(centroids)

    def register(self, centroid):
        logger.info(f"registering Person({self.next_id})")
        # register person as a trackable object
        self.objects[self.next_id] = Person(self.next_id, centroid)
        self.disappeared[self.next_id] = 0
        # update for next possible detection
        self.next_id += 1

    def deregister(self, person_id):
        logger.info(f"deregistering Person({person_id})")
        # If a person could not be recovered in a frame time of max_detections
        # then delete it from the records
        del self.objects[person_id]
        del self.disappeared[person_id]

    def update(self, input_centroids):
        # initialize trackable objects given input_centroids if Tracker is empty
        if len(self.objects) == 0 and len(input_centroids) != 0:
            logger.info("Initializing tracker")
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        # match input centroids to existing centroids
        object_ids = list(self.objects.keys())
        object_centroids = self._fetch_centroids()

        # no objects where detected, update accordingly
        if len(input_centroids) == 0:
            logger.debug("No faces were detected")
            for person_id in object_ids:
                self.disappeared[person_id] += 1

                if self.disappeared[person_id] > self.max_detections:
                    self.deregister(person_id)
            return self.objects

        # do so by minimizing euclidean distance between objects
        dist = distance.cdist(object_centroids, input_centroids)
        # follow PyImageSearch's algorithm
        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]

        # keep track of already examined rows and columns to determine which
        # objects must be updated, registered or deregistered
        used_rows = set()
        used_cols = set()
        for (row, col) in zip(rows, cols):
            # if row or col are already examined, skip them
            if row in used_rows or col in used_cols:
                continue

            # update matched object with new centroid
            person_id = object_ids[row]
            self.objects[person_id].centroid = input_centroids[col]
            self.disappeared[person_id] = 0
            # now rol and col are already examined
            used_rows.add(row)
            used_cols.add(col)

        # check unexamined rows and cols
        unused_rows = set(range(0, dist.shape[0])).difference(used_rows)
        unused_cols = set(range(0, dist.shape[1])).difference(used_cols)
        # in the event that there are more objects than detections, check
        # for possible disappeared objects
        if dist.shape[0] >= dist.shape[1]:
            for row in unused_rows:
                # grab person id given by row and update its disappearance
                person_id = object_ids[row]
                self.disappeared[person_id] += 1

                # if limit has been exceeded, then delete object
                if self.disappeared[person_id] > self.max_detections:
                    self.deregister(person_id)
        # otherwise, new objects have been introduced; register them
        else:
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


class Person(object):

    def __init__(self, id_, centroid):
        self.id_ = id_
        self.centroid = centroid
        self.is_wearing_mask = None
