# -*- coding: utf-8 -*-
# @Author  : Jiaxiang Shang
# @Email   : jiaxiang.shang@gmail.com
# @Time    : 8/11/20 8:30 PM


# system
from __future__ import print_function

import os
import sys

# python lib
import face_alignment
import numpy as np
import cv2

class LM_detector_howfar():
    def __init__(self, use_cnn_face_detector=True ,lm_type=2, device='cpu', face_detector='sfd'):
        if lm_type == 2:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D, device=device, flip_input=False, face_detector=face_detector)
        else:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._3D, device=device, flip_input=False, face_detector=face_detector)

    def lm_detection_howfar(self, image):
        """
        :param image: ndarray
        :return:
        lm: ndarray or None
        """
        # filter very large image
        scale = 1.0
        h, w, c = image.shape
        if max(h, w) > 900:
            scale = max(h, w) / (900.0)
            #image = sktrans.resize(image, [int(h/scale), int(w/scale), 3], anti_aliasing=True)
            image = cv2.resize(image, (int(w / scale), int(h / scale)))

        # torch
        detected_faces = self.fa.face_detector.detect_from_image(image[..., ::-1].copy())
        lm_howfar = self.fa.get_landmarks(image, detected_faces=detected_faces)

        # check the face detection bbox, that choose the largest one
        if lm_howfar is not None:
            list_hf = []
            list_size_detected_face = []
            for i in range(len(lm_howfar)):
                l_hf = lm_howfar[i]
                l_hf = l_hf * scale
                list_hf.append(l_hf)

                bbox = detected_faces[i]
                list_size_detected_face.append(bbox[2]-bbox[0] + bbox[3]-bbox[1])

            list_size_detected_face = np.array(list_size_detected_face)
            idx_max = np.argmax(list_size_detected_face)
            return list_hf[idx_max]
        else:
            return None