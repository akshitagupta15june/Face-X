import onnxruntime as ort
import numpy as np
import cv2
import datetime


class FaceRDA(object):
    def __init__(self, model_path, bfm=False):
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.bfm = bfm

    def __call__(self, img, roi_box):
        h, w = img.shape[:2]
        if self.bfm:
            image = cv2.resize(img, (120, 120))
        else:
            image = cv2.resize(img, (112, 112))
        input_data = ((image - 127.5) / 128).transpose((2, 0, 1))
        tensor = input_data[np.newaxis, :, :, :].astype("float32")
        begin = datetime.datetime.now()
        output = self.ort_session.run(None, {self.input_name: tensor})[0][0]
        end = datetime.datetime.now()
        print("facerda cpu times = ", end - begin)
        if self.bfm:
            vertices = self.decode_bfm(output, w, h, roi_box)
        else:
            vertices = self.decode(output, w, h, roi_box)
        return vertices

    def decode(self, output, w, h, roi_box):
        x1, x2, y1, y2 = w / 2, w / 2, -h / 2, h / 2
        v = np.array([[x1, 0, 0, x2],
                      [0, y1, 0, y2],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        vertices = v @ output
        sx, sy, ex, ey = roi_box
        vertices[0, :] = vertices[0, :] + sx
        vertices[1, :] = vertices[1, :] + sy
        return vertices

    def decode_bfm(self, output, w, h, roi_box):
        print(output.shape)
        # move to center of image
        output[0, :] = output[0, :] + 120 / 2
        output[1, :] = output[1, :] + 120 / 2
        # flip vertices along y-axis.
        output[1, :] = 120 - output[1, :] - 1
        vertices = output
        sx, sy, ex, ey = roi_box
        scale_x = (ex - sx) / 120
        scale_y = (ey - sy) / 120
        vertices[0, :] = vertices[0, :] * scale_x + sx
        vertices[1, :] = vertices[1, :] * scale_y + sy

        return vertices
