import os
import numpy as np
import torch.utils.data as td
import pandas as pd
import config
from csl_common.utils import geometry
from datasets.facedataset import FaceDataset

SUBSETS = ['pose', 'illumination', 'expression', 'make-up', 'occlusion', 'blur']


class WFLW(FaceDataset):

    NUM_LANDMARKS = 98
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))
    LANDMARKS_NO_OUTLINE = list(range(33,NUM_LANDMARKS))
    LANDMARKS_ONLY_OUTLINE = list(range(33))

    def __init__(self, root, cache_root=None, return_landmark_heatmaps=True, **kwargs):
        fullsize_img_dir=os.path.join(root, 'WFLW_images')
        super().__init__(root=root, cache_root=cache_root, fullsize_img_dir=fullsize_img_dir,
                         return_landmark_heatmaps=return_landmark_heatmaps, **kwargs)

    def _init(self):
        if not self.train:
            if self.test_split in SUBSETS:
                self.filter_labels({self.test_split:1})

    def parse_groundtruth_txt(self, gt_txt_file):
        num_lm_cols = self.NUM_LANDMARKS * 2
        columns_names = [
            'x',
            'y',
            'x2' ,
            'y2',
            'pose',
            'expression',
            'illumination',
            'make-up',
            'occlusion',
            'blur',
            'fname'
        ]
        ann = pd.read_csv(gt_txt_file,
                               header=None,
                               sep=' ',
                               usecols=range(num_lm_cols, num_lm_cols+11),
                               names=columns_names)
        ann['w'] = ann['x2'] - ann['x']
        ann['h'] = ann['y2'] - ann['y']

        landmarks = pd.read_csv(gt_txt_file,
                              header=None,
                              sep=' ',
                              usecols=range(0, num_lm_cols)).values

        ann['landmarks'] = [i for i in landmarks.reshape((-1, num_lm_cols//2, 2))]
        return ann

    def _load_annotations(self, split_name):
        split_name = 'train' if self.train else 'test'
        annotation_filename = os.path.join(self.cache_root, '{}_{}.pkl'.format(self.name, split_name))
        if os.path.isfile(annotation_filename):
            ann = pd.read_pickle(annotation_filename)
        else:
            print('Reading txt file...')
            gt_txt_file = os.path.join(self.root,
                                       'WFLW_annotations',
                                       'list_98pt_rect_attr_train_test',
                                       'list_98pt_rect_attr_'+split_name+'.txt')
            ann = self.parse_groundtruth_txt(gt_txt_file)
            ann.to_pickle(annotation_filename)
            print('done.')
        return ann

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        bb = [sample.x, sample.y, sample.x+sample.w, sample.y+sample.h]
        bb = geometry.extend_bbox(bb, db=0.1)
        face_id = int(sample.name)
        landmarks_for_crop = sample.landmarks.astype(np.float32) if self.crop_source == 'lm_ground_truth' else None
        return self.get_sample(sample.fname, landmarks_for_crop=landmarks_for_crop, bb=bb, id=face_id,
                               landmarks_to_return=sample.landmarks.astype(np.float32))


config.register_dataset(WFLW)


if __name__ == '__main__':
    from csl_common.utils.nn import Batch
    from csl_common.utils.common import init_random
    from csl_common.utils.ds_utils import build_transform
    from csl_common.vis import vis
    import config

    init_random(3)

    path = config.get_dataset_paths('wflw')[0]
    ds = WFLW(root=path, train=False, deterministic=True, use_cache=False, daug=0, image_size=256,
              transform=build_transform(deterministic=False, daug=0))
    ds.filter_labels({'pose': 1, 'occlusion':0, 'make-up':1})
    dl = td.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
    print(ds)

    for data in dl:
        batch = Batch(data, gpu=False)
        images = vis.to_disp_images(batch.images, denorm=True)
        # lms = lmutils.convert_landmarks(to_numpy(batch.landmarks), lmutils.LM98_TO_LM68)
        lms = batch.landmarks
        images = vis.add_landmarks_to_images(images, lms, draw_wireframe=False, color=(0,255,0), radius=3)
        vis.vis_square(images, nCols=10, fx=1., fy=1., normalize=False)
