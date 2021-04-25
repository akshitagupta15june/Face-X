import cv2
import numpy as np


class Data(object):
    def __init__(self, data, image_size, local_size):
        self.image_size = image_size
        self.local_size = local_size
        self.reset()
        self.img_files = data

    def __len__(self):
        return len(self.img_files)

    def reset(self):
        self.images = []
        self.points = []
        self.masks = []

    def flow(self, batch_size, hole_min=64, hole_max=128):
        np.random.shuffle(self.img_files)
        for i in self.img_files:
            img = cv2.imread(i)
            img = cv2.resize(img, self.image_size)[:, :, ::-1]
            self.images.append(img)

            x1 = np.random.randint(0, self.image_size[0] - self.local_size[0] + 1)
            y1 = np.random.randint(0, self.image_size[1] - self.local_size[1] + 1)
            x2, y2 = np.array([x1, y1]) + np.array(self.local_size)
            self.points.append([x1, y1, x2, y2])

            w, h = np.random.randint(hole_min, hole_max, 2)
            p1 = x1 + np.random.randint(0, self.local_size[0] - w)
            q1 = y1 + np.random.randint(0, self.local_size[1] - h)
            p2 = p1 + w
            q2 = q1 + h

            m = np.zeros((self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
            m[q1:q2 + 1, p1:p2 + 1] = 1
            self.masks.append(m)

            if len(self.images) == batch_size:
                inputs = np.asarray(self.images, dtype=np.float32) / 255
                points = np.asarray(self.points, dtype=np.int32)
                masks = np.asarray(self.masks, dtype=np.float32)
                self.reset()
                yield inputs, points, masks

