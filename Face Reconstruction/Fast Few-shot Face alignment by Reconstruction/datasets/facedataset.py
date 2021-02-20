import numpy as np

from datasets.imagedataset import ImageDataset
from landmarks import lmutils


class FaceDataset(ImageDataset):
    NUM_LANDMARKS = 68
    LANDMARKS_ONLY_OUTLINE = list(range(17))
    LANDMARKS_NO_OUTLINE = list(range(17,NUM_LANDMARKS))
    ALL_LANDMARKS =  LANDMARKS_ONLY_OUTLINE + LANDMARKS_NO_OUTLINE

    def __init__(self, return_landmark_heatmaps=False, landmark_sigma=9, align_face_orientation=False, **kwargs):
        super().__init__(**kwargs)
        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.landmark_sigma = landmark_sigma
        self.empty_landmarks = np.zeros((self.NUM_LANDMARKS, 2), dtype=np.float32)
        self.align_face_orientation = align_face_orientation

    @staticmethod
    def _get_expression(sample):
        return np.array([[0,0,0]], dtype=np.float32)

    @staticmethod
    def _get_identity(sample):
        return -1

    def _crop_landmarks(self, lms):
         return self.loader._cropper.apply_to_landmarks(lms)[0]

    def get_sample(self, filename, bb=None, landmarks_for_crop=None, id=None, landmarks_to_return=None):
        try:
            crop_mode = 'landmarks' if landmarks_for_crop is not None else 'bounding_box'
            crop_params = {'landmarks': landmarks_for_crop,
                           'bb': bb,
                           'id': id,
                           'aligned': self.align_face_orientation,
                           'mode': crop_mode}
            image = self.loader.load_crop(filename, **crop_params)
        except:
            print('Could not load image {}'.format(filename))
            raise

        relative_landmarks = self._crop_landmarks(landmarks_to_return) \
            if landmarks_to_return is not None else self.empty_landmarks

        # self.show_landmarks(image, landmarks)

        sample = {'image': image,
                  'landmarks': relative_landmarks,
                  'pose': np.zeros(3, dtype=np.float32)}

        if self.transform is not None:
            sample = self.transform(sample)
        target = self.target_transform(sample) if self.target_transform else None

        # self.show_landmarks(sample['image'], sample['landmarks'])

        if self.crop_type != 'fullsize':
            sample = self.crop_to_tensor(sample)
            if target is not None:
                target = self.crop_to_tensor(target)

        sample.update({
            'fnames': filename,
            'bb': bb if bb is not None else [0,0,0,0],
            # 'expression':self._get_expression(sample),
            # 'id': self._get_identity(sample),
        })

        if target is not None:
            sample['target'] = target

        if self.return_landmark_heatmaps and self.crop_type != 'fullsize':
            from landmarks import lmconfig as lmcfg
            heatmap_size = lmcfg.HEATMAP_SIZE
            scaled_landmarks = sample['landmarks'] * (heatmap_size / self.image_size)
            sample['lm_heatmaps'] = lmutils.create_landmark_heatmaps(scaled_landmarks, self.landmark_sigma,
                                                                     self.ALL_LANDMARKS, heatmap_size)
        return sample


    def show_landmarks(self, img, landmarks):
        import cv2
        for lm in landmarks:
            lm_x, lm_y = lm[0], lm[1]
            cv2.circle(img, (int(lm_x), int(lm_y)), 3, (0, 0, 255), -1)
        cv2.imshow('landmarks', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)