import os

import numpy as np
import pandas as pd

from torchvision import transforms as tf

import csl_common.utils.transforms as csl_tf
from csl_common.utils.io_utils import makedirs

# To avoid exceptions when loading truncated image files
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_openface_detection(lmFilepath, numpy_lmFilepath=None, from_sequence=False, use_cache=True,
                            return_num_faces=False, expected_face_center=None):
    num_faces_in_image = 0
    try:
        if numpy_lmFilepath is not None:
            npfile = numpy_lmFilepath + '.npz'
        else:
            npfile = lmFilepath + '.npz'
        if os.path.isfile(npfile) and use_cache:
            try:
                data = np.load(npfile)
                of_conf, landmarks, pose = [data[arr] for arr in data.files]
                if of_conf > 0:
                    num_faces_in_image = 1
            except:
                print('Could not open file {}'.format(npfile))
                raise
        else:
            if from_sequence:
                lmFilepath = lmFilepath.replace('features', 'features_sequence')
                lmDir, fname = os.path.split(lmFilepath)
                clip_name = os.path.split(lmDir)[1]
                lmFilepath = os.path.join(lmDir, clip_name)
                features = pd.read_csv(lmFilepath + '.csv', skipinitialspace=True)
                frame_num = int(os.path.splitext(fname)[0])
                features = features[features.frame == frame_num]
            else:
                features = pd.read_csv(lmFilepath + '.csv', skipinitialspace=True)
            features.sort_values('confidence', ascending=False, inplace=True)
            selected_face_id = 0
            num_faces_in_image = len(features)
            if num_faces_in_image > 1 and expected_face_center is not None:
                max_face_size = 0
                min_distance = 1000
                for fid in range(len(features)):
                    face = features.iloc[fid]
                    # if face.confidence < 0.2:
                    #     continue
                    landmarks_x = face.as_matrix(columns=['x_{}'.format(i) for i in range(68)])
                    landmarks_y = face.as_matrix(columns=['y_{}'.format(i) for i in range(68)])

                    landmarks = np.vstack((landmarks_x, landmarks_y)).T
                    face_center = landmarks.mean(axis=0)
                    distance = ((face_center - expected_face_center)**2).sum()**0.5
                    if distance < min_distance:
                        min_distance = distance
                        selected_face_id = fid

            try:
                face = features.iloc[selected_face_id]
            except KeyError:
                face = features
            of_conf = face.confidence
            landmarks_x = face.as_matrix(columns=['x_{}'.format(i) for i in range(68)])
            landmarks_y = face.as_matrix(columns=['y_{}'.format(i) for i in range(68)])
            landmarks = np.vstack((landmarks_x, landmarks_y)).T
            pitch = face.pose_Rx
            yaw = face.pose_Ry
            roll = face.pose_Rz
            pose = np.array((pitch, yaw, roll), dtype=np.float32)
            if numpy_lmFilepath is not None:
                makedirs(npfile)
            np.savez(npfile, of_conf, landmarks, pose)
    except IOError as e:
        # raise IOError("\tError: Could not load landmarks from file {}!".format(lmFilepath))
        # pass
        # print(e)
        of_conf = 0
        landmarks = np.zeros((68,2), dtype=np.float32)
        pose = np.zeros(3, dtype=np.float32)

    result = [of_conf, landmarks.astype(np.float32), pose]
    if return_num_faces:
        result += [num_faces_in_image]
    return result


def build_transform(deterministic, color=True, daug=0):
    transforms = []
    if not deterministic:
        transforms = [csl_tf.RandomHorizontalFlip(0.5)]
        if daug == 1:
            transforms += [csl_tf.RandomAffine(3, translate=[0.025, 0.025], scale=[0.975, 1.025], shear=0, keep_aspect=False)]
        elif daug == 2:
            transforms += [csl_tf.RandomAffine(3, translate=[0.035, 0.035], scale=[0.970, 1.030], shear=2, keep_aspect=False)]
        elif daug == 3:
            transforms += [csl_tf.RandomAffine(20, translate=[0.035, 0.035], scale=[0.970, 1.030], shear=0, keep_aspect=False)]
        elif daug == 4:
            transforms += [csl_tf.RandomAffine(45, translate=[0.035, 0.035], scale=[0.940, 1.030], shear=5, keep_aspect=False)]
        elif daug == 5:
            transforms += [csl_tf.RandomAffine(60, translate=[0.035, 0.035], scale=[0.940, 1.030], shear=5, keep_aspect=False)]
        elif daug == 6:  # CVPR landmark training
            transforms += [csl_tf.RandomAffine(30, translate=[0.04, 0.04], scale=[0.940, 1.050], shear=5, keep_aspect=False)]
        elif daug == 7:
            transforms += [csl_tf.RandomAffine(0, translate=[0.04, 0.04], scale=[0.940, 1.050], shear=5, keep_aspect=False)]
    return tf.Compose(transforms)

