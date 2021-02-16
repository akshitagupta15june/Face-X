import os
import numpy as np
import pandas as pd
import torch.utils.data as td
from csl_common.vis import vis
from csl_common.utils import geometry
from datasets import facedataset


class AFLW(facedataset.FaceDataset):

    NUM_LANDMARKS = 19
    ALL_LANDMARKS = list(range(NUM_LANDMARKS))
    LANDMARKS_NO_OUTLINE = ALL_LANDMARKS    # no outlines in AFLW
    LANDMARKS_ONLY_OUTLINE = ALL_LANDMARKS  # no outlines in AFLW

    def __init__(self, root, cache_root=None, test_split='full', landmark_ids=range(19), **kwargs):

        assert test_split in ['full', 'frontal']
        fullsize_img_dir = os.path.join(root, 'data/flickr')

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=fullsize_img_dir,
                         test_split=test_split,
                         landmark_ids=landmark_ids,
                         **kwargs)

    @property
    def labels(self):
        return self.annotations.ID.values

    @property
    def heights(self):
        return self.annotations.face_h.values

    @property
    def widths(self):
        return self.annotations.face_w.values

    def _load_annotations(self, split):
        annotation_filename = os.path.join(self.cache_root, 'alfw.pkl')
        self.annotations_original = pd.read_pickle(annotation_filename)
        print("Number of images: {}".format(len(self.annotations_original)))
        self.frontal_only = split == 'frontal'
        return self.make_split(self.train, self.frontal_only)

    def make_split(self, train, only_frontal):
        import scipy.io
        # Additional annotations from http://mmlab.ie.cuhk.edu.hk/projects/compositional.html
        annots = scipy.io.loadmat(os.path.join(self.cache_root, 'AFLWinfo_release.mat'))

        train_ids, test_ids = annots['ra'][0][:20000] - 1, annots['ra'][0][20000:] - 1
        ids = annots['ra'][0] - 1

        # merge original and additional annotations
        lms = annots['data'][ids]
        lms = np.dstack((lms[:,:19], lms[:, 19:]))
        lms_list = [l for l in lms]
        mask_new = annots['mask_new'][ids]

        bbox = annots['bbox'][ids]
        x1, x2, y1, y2 = bbox[:,0], bbox[:,1], bbox[:, 2], bbox[:, 3]
        fnames = [f[0][0] for f in annots['nameList'][ids]]
        annotations_additional = pd.DataFrame({
            'fname':fnames,
            'ra': ids,
            'landmarks_full':lms_list,
            'masks': [m for m in mask_new],
            'face_x': x1,
            'face_y': y1,
            'face_w': x2 - x1,
            'face_h': y2 - y1
        })

        ad = annotations_additional
        ao = self.annotations_original

        # self.annotations_test = self.annotations_original[self.annotations.fname.isin(fnames)]
        pd.set_option('display.expand_frame_repr', False)
        merge_on=['fname', 'face_x', 'face_y', 'face_w', 'face_h']
        annotations = pd.merge(ad, ao, on=merge_on)
        annotations = annotations.sort_values('ra')

        split_ids = train_ids if train else test_ids
        annotations = annotations[annotations.ra.isin(split_ids)]

        if not train and only_frontal:
            mask_all_lms_visible = np.stack(annotations.masks.values).min(axis=1) == 1
            annotations = annotations[mask_all_lms_visible]
            print(len(annotations))

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        face_id = sample.ra
        bb = [sample.face_x, sample.face_y, sample.face_x+sample.face_w, sample.face_y+sample.face_h]
        landmarks = sample.landmarks_full.astype(np.float32)
        landmarks_for_crop = landmarks if self.crop_source == 'lm_ground_truth' else None
        return self.get_sample(sample.fname, bb, landmarks_for_crop=landmarks_for_crop,  id=face_id,
                               landmarks_to_return=landmarks)


import config
config.register_dataset(AFLW)

if __name__ == '__main__':

    from csl_common.utils.nn import Batch, denormalize
    import utils.common

    utils.common.init_random()

    ds = AFLW(train=True, deterministic=True, use_cache=True, image_size=256)
    dl = td.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)

    for data in dl:
        batch = Batch(data, gpu=False)
        inputs = batch.images.clone()
        denormalize(inputs)
        imgs = vis.add_landmarks_to_images(inputs.numpy(), batch.landmarks.numpy(), radius=3, color=(0,255,0))
        print(batch.fnames)
        vis.vis_square(imgs, nCols=10, fx=1.0, fy=1.0, normalize=False)
