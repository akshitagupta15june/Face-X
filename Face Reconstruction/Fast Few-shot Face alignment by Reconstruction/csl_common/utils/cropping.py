import cv2
import numpy as np
import matplotlib.pyplot as plt
import numbers

from csl_common.utils import geometry
from skimage import exposure


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class FaceCrop():
    def __init__(
            self,
            img,
            output_size,
            bbox=None,
            landmarks=None,
            img_already_cropped=False,  # -> only crops landmarks
            crop_by_eye_mouth_dist=False,
            crop_by_height=True,
            align_face_orientation=False,
            align_eyes_horizontally=False,
            scale=1.0,
            keep_aspect=True,
            crop_move_top_factor=0.25,       # move top by factor of face height in respect to eye brows
            crop_move_bottom_factor=0.15     # move bottom by factor of face height in respect to chin bottom point
    ):
        if crop_by_eye_mouth_dist and crop_by_height:
            raise ValueError("'crop_by_eye_mouth_dist' and 'crop_by_height' cannot both be true. "
                             "Default: crop_by_height=True.")
        if isinstance(output_size, numbers.Number):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.align_face_orientation = align_face_orientation
        self.img_already_cropped = img_already_cropped
        self.keep_aspect = keep_aspect
        self.crop_by_eye_mouth_dist = crop_by_eye_mouth_dist
        self.crop_by_height = crop_by_height
        self.M = None
        self.tl = None
        self.br = None
        self.scale = scale
        self.lms = landmarks
        self.img = img
        self.angle_x_deg = 0
        self.crop_move_top_factor = crop_move_top_factor
        self.crop_move_bottom_factor = crop_move_bottom_factor
        self.align_eyes_horizontally = align_eyes_horizontally

        if landmarks is None and bbox is None:
            bbox = np.array([0, 0, img.shape[1]-1, img.shape[0]-1])

        if bbox is not None:
            bbox = np.asarray(bbox)
            h,w = bbox[2:] - bbox[:2]
            if h < 10 or w < 10:
                bbox = np.array([0, 0, img.shape[1]-1, img.shape[0]-1])
            self.tl = bbox[:2].astype(int)
            self.br = bbox[2:4].astype(int)

        if landmarks is not None:
            if len(landmarks) == 5:
                self.crop_by_eye_mouth_dist = True
            self.calculate_crop_parameters(img, landmarks)

        assert bbox is not None or landmarks is not None
        assert self.tl is not None and self.br is not None
        assert not self.align_face_orientation or self.M is not None

    def __get_eye_coordinates(self, lms):
        if lms.shape[0] == 68:
            id_eye_l, id_eye_r = [36, 39],  [42, 45]
        elif lms.shape[0] == 98:
            id_eye_l, id_eye_r = [60, 64],  [68, 72]
        elif lms.shape[0] == 37:
            id_eye_l, id_eye_r = [13, 16],  [19, 22]
        elif lms.shape[0] == 21:
            id_eye_l, id_eye_r = [6, 8],  [9, 11]
        elif lms.shape[0] == 4:
            id_eye_l, id_eye_r = [0],  [1]
        elif lms.shape[0] == 5:
            # id_eye_l, id_eye_r = [1],  [2]
            id_eye_l, id_eye_r = [0],  [1]
        else:
            raise ValueError("Invalid landmark format!")
        eye_l, eye_r = lms[id_eye_l].mean(axis=0),  lms[id_eye_r].mean(axis=0)  # eye centers
        if eye_r[0] < eye_l[0]:
            eye_r, eye_l = eye_l, eye_r
        return eye_l, eye_r

    def __get_mouth_coordinates(self, lms):
        if lms.shape[0] == 68:
            id_mouth_l, id_mouth_r = 48,  54
        elif lms.shape[0] == 98:
            id_mouth_l, id_mouth_r = 76,  82
        elif lms.shape[0] == 37:
            id_mouth_l, id_mouth_r = 25,  31
        elif lms.shape[0] == 21:
            id_mouth_l, id_mouth_r = 17,  19
        elif lms.shape[0] == 4:
            id_mouth_l, id_mouth_r = 2,  3
        elif lms.shape[0] == 5:
            id_mouth_l, id_mouth_r = 3, 4
        else:
            raise ValueError("Invalid landmark format!")
        return lms[id_mouth_l],  lms[id_mouth_r]  # outer landmarks

    def __get_chin_coordinates(self, lms):
        if lms.shape[0] == 68:
            id_chin = 8
        elif lms.shape[0] == 98:
            id_chin = 16
        else:
            raise ValueError("Invalid landmark format!")
        return lms[id_chin]

    def __get_face_center(self, lms, return_scale=False):
        eye_l, eye_r = self.__get_eye_coordinates(lms)
        mouth_l, mouth_r = self.__get_mouth_coordinates(lms)
        chin = self.__get_chin_coordinates(lms)
        eye_c = (eye_l+eye_r)/2
        # vec_nose = eye_c - (mouth_l+mouth_r)/2
        vec_nose = eye_c - chin
        c = eye_c - 0.25*vec_nose
        if return_scale:
            s = int(0.9*np.linalg.norm(vec_nose))
            return c, s
        else:
            return c

    def __calc_rotation_matrix(self, center, eye_l, eye_r, nose_upper, chin):
        def calc_angles_deg(vec):
            vnorm = vec / np.linalg.norm(vec)
            angle_x = np.arcsin(vnorm[1])
            angle_y = np.arcsin(vnorm[0])
            return np.rad2deg(angle_x), np.rad2deg(angle_y)

        vec_eye = eye_r - eye_l
        vec_nose = nose_upper - chin
        w = np.linalg.norm(vec_eye)
        h = np.linalg.norm(vec_nose)
        vx = vec_eye / w

        is_profile_face = w < h * 0.2
        if is_profile_face:
            # use orientation of nose
            ax, ay = calc_angles_deg(vec_nose)
            self.angle_x_deg = ay
        else:
            # use orientation of eye
            angle_x = np.arcsin(vx[1])
            self.angle_x_deg = np.rad2deg(angle_x)
        return cv2.getRotationMatrix2D(tuple(center), self.angle_x_deg, 1.0)

    def __rotate_image(self, img, M):
        return cv2.warpAffine(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)

    def __rotate_landmarks(self, lms, M):
        _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
        return M.dot(_lms_hom.T).T  # apply transformation

    def calculate_crop_parameters(self, img, lms_orig):
        self.__img_shape = img.shape
        lms = lms_orig.copy()

        self.face_center, self.face_scale = self.__get_face_center(lms, return_scale=True)

        nonzeros= lms[:,0] > 0

        if self.align_face_orientation:
            eye_l, eye_r = self.__get_eye_coordinates(lms)
            chin = self.__get_chin_coordinates(lms)
            nose_upper = lms[51]
            self.M = self.__calc_rotation_matrix(self.face_center, eye_l, eye_r, nose_upper, chin)
            lms = self.__rotate_landmarks(lms, self.M)

        if self.align_eyes_horizontally:
            cx = (lms[nonzeros,0].min()+lms[nonzeros,0].max())/2
            self.face_center[0] = cx

        if self.crop_by_eye_mouth_dist:
            self.tl = (self.face_center - self.face_scale).astype(int)
            self.br = (self.face_center + self.face_scale).astype(int)
        elif self.crop_by_height:
            # calc height
            t = lms[nonzeros, 1].min()
            b = lms[nonzeros, 1].max()
            if t > b:
                t, b = b, t

            h = b - t
            assert(h >= 0)

            # calc width
            l = lms[nonzeros, 0].min()
            r = lms[nonzeros, 0].max()
            if l > r:
                l, r = r, l
            w = r - l
            assert(w >= 0)

            has_eyebrows = len(lms) != 68 or len(lms) != 21 and len(lms) != 98
            if not has_eyebrows:
                # height is
                h *= 1.5
                t = t - h/2
                b = b + h/2

            # enlarge a little
            min_row, max_row = int(t - self.crop_move_top_factor * h), \
                               int(b + self.crop_move_bottom_factor * h)

            # calc new width
            if self.keep_aspect:
                s = (max_row - min_row)/2
                min_col, max_col = self.face_center[0] - s, self.face_center[0] + s
            else:
                min_col, max_col = int(l - 0.15 * w), int(r + 0.15 * w)

            # in case right eye is actually left of right eye...
            if min_col > max_col:
                min_col, max_col = min_col, max_col

            # crop = img[int(min_row):int(max_row), int(min_col):int(max_col)]
            # plt.imshow(crop)
            # plt.show()

            self.tl = np.array((min_col, min_row))
            self.br = np.array((max_col, max_row))
        else:
            raise ValueError

        # extend image area so crops can be fully rotated for data augmentations (= set size to length of diagonal)
        scale_factor = 2**0.5
        bbox = np.concatenate((self.tl, self.br))
        bbox_crop = geometry.scaleBB(bbox, scale_factor, scale_factor, typeBB=2)
        self.tl = bbox_crop[0:2].astype(int)
        self.br = bbox_crop[2:4].astype(int)


    def apply_to_image(self, img=None, with_hist_norm=False, border_mode='black'):
        if img is None:
            img = self.img

        if self.img_already_cropped:
            h, w = img.shape[:2]
            if (w,h) != self.output_size:
                img = cv2.resize(img, self.output_size, interpolation=cv2.INTER_CUBIC)
            return img

        h,w = img.shape[:2]
        if border_mode == 'edge':
            img_padded = np.pad(img, ((h,h), (w,w),(0,0)), mode='edge')
        elif border_mode == 'black':
            img_padded = np.pad(img, ((h,h), (w,w),(0,0)), mode='constant')
        else:
            img_padded = np.pad(img, ((h,h), (w,w),(0,0)), mode='symmetric')
            s = int(img.shape[0]*0.2)
            img_padded = cv2.blur(img_padded, (s,s))
            img_padded[h:2*h,w:2*w] = img

        tl_padded = self.tl + (w,h)
        br_padded = self.br + (w,h)

        # extend image in case padded image is still too small
        dilate = -np.minimum(tl_padded, 0)
        padding = [
            (dilate[1], dilate[1]),
            (dilate[0], dilate[0]),
             (0,0)
        ]
        try:
            img_padded = np.pad(img_padded, padding, 'constant')
        except TypeError:
            plt.imshow(img)
            plt.show()
        tl_padded += dilate
        br_padded += dilate

        # rotate image
        if self.align_face_orientation and self.lms is not None:
            face_center = self.__get_face_center(self.lms, return_scale=False)
            M  = cv2.getRotationMatrix2D(tuple(face_center+(w,h)), self.angle_x_deg, 1.0)
            img_padded = self.__rotate_image(img_padded, M)

        crop = img_padded[tl_padded[1]: br_padded[1], tl_padded[0]: br_padded[0]]

        try:
            resized_crop = cv2.resize(crop, self.output_size, interpolation=cv2.INTER_CUBIC)
        except cv2.error:
            print('img size', img.shape)
            print(self.tl)
            print(self.br)
            print('dilate: ', dilate)
            print('padding: ', padding)
            print('img pad size', img_padded.shape)
            print(tl_padded)
            print(br_padded)
            plt.imshow(img_padded)
            plt.show()
            raise

        # optional image normalization
        if with_hist_norm:
            p2, p98 = np.percentile(crop, (2, 98))
            resized_crop = exposure.rescale_intensity(resized_crop, in_range=(p2, p98))

        return np.clip(resized_crop, 0, 255)

    def apply_to_landmarks(self, lms_orig, pose=None):
        if lms_orig is None:
            return lms_orig, pose

        if pose is not None:
            pose_new = np.array(pose).copy()
        else:
            pose_new = None

        lms = lms_orig.copy()

        if self.align_face_orientation and self.lms is not None:
            # rotate landmarks
            # if not self.img_already_cropped:
            #     lms[:,0] += self.img.shape[1]
            #     lms[:,1] += self.img.shape[0]
            self.face_center_orig = self.__get_face_center(self.lms, return_scale=False)
            M = cv2.getRotationMatrix2D(tuple(self.face_center_orig), self.angle_x_deg, 1.0)
            lms = self.__rotate_landmarks(lms, M).astype(np.float32)

            # if not self.img_already_cropped:
            #     lms[:,0] -= self.img.shape[1]
            #     lms[:,1] -= self.img.shape[0]

            if pose_new is not None:
                pose_new[2] = 0.0

        # tl = (self.face_center - self.face_scale).astype(int)
        # br = (self.face_center + self.face_scale).astype(int)

        tl = self.tl
        br = self.br
        crop_width = br[0] - tl[0]
        crop_height = br[1] - tl[1]

        lms_new = lms.copy()
        lms_new[:, 0] = (lms_new[:, 0] - tl[0]) * self.output_size[0] / crop_width
        lms_new[:, 1] = (lms_new[:, 1] - tl[1]) * self.output_size[1] / crop_height

        return lms_new, pose_new

    def apply_to_landmarks_inv(self, lms):
        tl = self.tl
        br = self.br
        crop_width = br[0] - tl[0]
        crop_height = br[1] - tl[1]

        lms_new = lms.copy()
        lms_new[:, 0] = lms_new[:, 0] * (crop_width / self.output_size[0]) + tl[0]
        lms_new[:, 1] = lms_new[:, 1] * (crop_height / self.output_size[1]) + tl[1]

        # return lms_new
        if self.M is not None:
            M = cv2.getRotationMatrix2D(tuple(self.face_center_orig), -self.angle_x_deg, 1.0)
            lms_new = self.__rotate_landmarks(lms_new, M)

        return lms_new


