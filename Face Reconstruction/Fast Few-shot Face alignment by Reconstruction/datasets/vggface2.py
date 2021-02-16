import os
import time
import numpy as np
import torch.utils.data as td
import pandas as pd

from csl_common.utils import log, cropping, ds_utils, geometry
from csl_common.vis import vis
from datasets import facedataset
import config as cfg


class VggFace2(facedataset.FaceDataset):

    meta_folder = 'bb_landmark'
    image_folder = 'imgs'

    def __init__(self, root, cache_root=None, train=True, crop_source='bb_ground_truth',
                 return_modified_images=False, min_face_height=100, **kwargs):

        assert(crop_source in ['bb_ground_truth', 'lm_ground_truth', 'lm_openface'])

        self.split_folder = 'train' if train else 'test'
        fullsize_img_dir = os.path.join(root, self.split_folder, self.image_folder)
        self.annotation_filename = 'loose_bb_{}.csv'.format(self.split_folder)

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=fullsize_img_dir,
                         crop_source=crop_source,
                         crop_dir=os.path.join(self.split_folder, 'crops'),
                         return_landmark_heatmaps=False,
                         return_modified_images=return_modified_images,
                         **kwargs)

        self.min_face_height = min_face_height

        # shuffle images since dataset is sorted by identities
        import sklearn.utils
        self.annotations = sklearn.utils.shuffle(self.annotations)

        print("Removing faces with height <= {:.2f}px...".format(self.min_face_height))
        self.annotations = self.annotations[self.annotations.H > self.min_face_height]
        print("Number of images: {}".format(len(self)))
        print("Number of identities: {}".format(self.annotations.ID.nunique()))


    @property
    def cropped_img_dir(self):
        return os.path.join(self.cache_root, self.split_folder, 'crops', self.crop_source)

    def get_crop_extend_factors(self):
        return 0.05, 0.1

    @property
    def ann_csv_file(self):
        return os.path.join(self.root, self.meta_folder, self.annotation_filename)

    def _read_annots_from_csv(self):
        print('Reading CSV file...')
        annotations = pd.read_csv(self.ann_csv_file)
        print(f'{len(annotations)} lines read.')

        # assign new continuous ids to persons (0, range(n))
        print("Creating id labels...")
        _ids = annotations.NAME_ID
        _ids = _ids.map(lambda x: int(x.split('/')[0][1:]))
        annotations['ID'] = _ids

        return annotations

    def _load_annotations(self, split):
        path_annotations_mod = os.path.join(self.cache_root, self.annotation_filename + '.mod_full.pkl')
        if os.path.isfile(path_annotations_mod):
            annotations = pd.read_pickle(path_annotations_mod)
        else:
            annotations = self._read_annots_from_csv()
            annotations.to_pickle(path_annotations_mod)
        return annotations

    @property
    def labels(self):
        return self.annotations.ID.values

    @property
    def heights(self):
        return self.annotations.H.values

    @property
    def widths(self):
        return self.annotations.W.values

    @staticmethod
    def _get_identity(sample):
        return sample.ID

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        # bb = self.get_adjusted_bounding_box(sample.X, sample.Y, sample.W, sample.H)
        bb = [sample.X, sample.Y, sample.X+sample.W, sample.Y+sample.H]
        bb = geometry.extend_bbox(bb, dt=0.05, db=0.10)
        landmarks_for_crop = sample.landmarks.astype(np.float32) if self.crop_source == 'lm_ground_truth' else None
        return self.get_sample(sample.NAME_ID+'.jpg', bb, landmarks_for_crop)


cfg.register_dataset(VggFace2)


def extract_features(split, st=None, nd=None):
    """ Extract facial features (landmarks, pose,...) from images """
    import glob
    assert(split in ['train', 'test'])
    person_dirs = sorted(glob.glob(os.path.join(VGGFACE2_ROOT, split, 'imgs', '*')))[st:nd]
    # print(os.path.join(cfg.VGGFACE2_ROOT, split, 'imgs', '*'))
    for cnt, img_dir in enumerate(person_dirs):
        folder_name = os.path.split(img_dir)[1]
        out_dir = os.path.join(VGGFACE2_ROOT_LOCAL, split, 'features', folder_name)
        log.info("{}/{}".format(cnt, len(person_dirs)))
        cropping.run_open_face(img_dir, out_dir, is_sequence=False)


def extract_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--st', default=None, type=int)
    parser.add_argument('--nd', default=None, type=int)
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    extract_features(args.split, st=args.st, nd=args.nd)


if __name__ == '__main__':
    # extract_main()
    # exit()
    from utils.nn import Batch
    import utils.common as util
    util.init_random()

    ds = VggFace2(train=True, deterministic=True, use_cache=False, align_face_orientation=False,
                  return_modified_images=False, image_size=256)
    micro_batch_loader = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    f = 1.0
    t = time.perf_counter()
    for iter, data in enumerate(micro_batch_loader):
        print('t load:', time.perf_counter() - t)
        t = time.perf_counter()
        batch = Batch(data, gpu=False)
        print('t Batch:', time.perf_counter() - t)
        images = ds_utils.denormalized(batch.images)
        vis.vis_square(images, fx=f, fy=f, normalize=False, nCols=10, wait=0)
