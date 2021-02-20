import os
import numpy as np
import torch.utils.data as td
import pandas as pd

import config
from csl_common.utils.nn import Batch
from csl_common.utils import geometry
from datasets import facedataset


def read_300W_detection(lmFilepath):
    lms = []
    with open(lmFilepath) as f:
        for line in f:
            try:
                x,y = [float(e) for e in line.split()]
                lms.append((x, y))
            except:
                pass
    assert(len(lms) == 68)
    landmarks = np.vstack(lms)
    return landmarks


class W300(facedataset.FaceDataset):

    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth']

    NUM_LANDMARKS = 68
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))
    LANDMARKS_NO_OUTLINE = list(range(17,68))
    LANDMARKS_ONLY_OUTLINE = list(range(17))

    def __init__(self, root, cache_root=None, train=True, test_split='full',
                 crop_source='bb_detector', return_landmark_heatmaps=False,
                 return_modified_images=False, **kwargs):

        test_split = test_split.lower()
        if not train:
            assert(test_split in ['train', 'common', 'challenging', '300w', 'full'])
        assert(crop_source in W300.CROP_SOURCES)

        self.bounding_box_dir = os.path.join(root, 'Bounding Boxes')

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=os.path.join(root, 'images'),
                         train=train,
                         test_split=test_split,
                         crop_source=crop_source,
                         return_landmark_heatmaps=return_landmark_heatmaps,
                         return_modified_images=return_modified_images,
                         **kwargs)


        if self.crop_type == 'fullsize':
            self.transform = lambda x:x


    def _load_annotations(self, split):
        import scipy.io
        import glob

        split_defs = {
            'train': [
                ('train/afw', 'afw'),
                ('train/helen', 'helen_trainset'),
                ('train/lfpw', 'lfpw_trainset')
            ],
            'common': [
                ('test/common/helen', 'helen_testset'),
                ('test/common/lfpw', 'lfpw_testset')
            ],
            'challenging': [
                ('test/challenging/ibug', 'ibug')
            ],
            'full': [
                ('test/common/helen', 'helen_testset'),
                ('test/common/lfpw', 'lfpw_testset'),
                ('test/challenging/ibug', 'ibug')
            ],
            '300w': [
                ('test/300W/01_Indoor', None),
                ('test/300W/01_Outdoor', None)
            ]
        }

        ann = []

        bboxes = []
        for id, subset in enumerate(split_defs[split]):

            im_dir, bbox_file_suffix = subset

            # get image file paths and read GT landmarks
            ext = "*.jpg"
            if 'lfpw' in im_dir or '300W' in im_dir:
                ext = "*.png"
            for img_file in sorted(glob.glob(os.path.join(self.fullsize_img_dir, im_dir, ext))):

                path_abs_noext = os.path.splitext(img_file)[0]
                filename_noext =  os.path.split(path_abs_noext)[1]
                filename = os.path.split(img_file)[1]
                path_rel = os.path.join(im_dir, filename)

                # load landmarks from *.pts files
                landmarks = read_300W_detection(path_abs_noext+'.pts')
                ann.append({'imgName': str(filename), 'fname': path_rel, 'landmarks': landmarks})

            # load supplied detected bounding boxes from MAT file
            if bbox_file_suffix is not None:
                mat_file = os.path.join(self.bounding_box_dir, 'bounding_boxes_{}.mat'.format(bbox_file_suffix))
                subset_bboxes = scipy.io.loadmat(mat_file)
                for item in subset_bboxes['bounding_boxes'][0]:
                    imgName, bb_detector, bb_ground_truth = item[0][0]
                    bboxes.append({'imgName': str(imgName[0]),
                                   'bb_detector': bb_detector[0],
                                   'bb_ground_truth': bb_ground_truth[0]})

        annotations = pd.DataFrame(ann)
        if len(bboxes) > 0:
            df_bboxes = pd.DataFrame(bboxes)
            annotations = annotations.merge(df_bboxes, on='imgName', how='left')

        return annotations

    @property
    def labels(self):
        return None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        bb = sample.bb_detector if self.crop_source == 'bb_detector' else sample.bb_ground_truth
        bb = geometry.extend_bbox(bb, dt=0.2, db=0.12)
        landmarks =  sample.landmarks.astype(np.float32)
        landmarks_for_crop = None
        if self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = landmarks
        return self.get_sample(sample.fname, bb, landmarks_for_crop, landmarks_to_return=landmarks)


config.register_dataset(W300)


if __name__ == '__main__':
    from csl_common.vis import vis
    import torch
    import config

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dirs = config.get_dataset_paths('w300')
    ds = W300(root=dirs[0], cache_root=dirs[1], train=False, deterministic=True, use_cache=False, image_size=256,
              test_split='challenging', daug=0, align_face_orientation=True, crop_source='lm_ground_truth')
    dl = td.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)

    for data in dl:
        batch = Batch(data, gpu=False)
        inputs = batch.images.clone()
        imgs = vis.to_disp_images(inputs, denorm=True)
        imgs = vis.add_landmarks_to_images(imgs, batch.landmarks, radius=3, color=(0,255,0))
        # imgs = vis.add_landmarks_to_images(imgs, data['landmarks_of'].numpy(), color=(1,0,0))
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)