import os
import numpy as np

import torch.utils.data as td
import pandas as pd

from csl_common.utils import geometry
from csl_common.utils import ds_utils
from csl_common.utils.nn import to_numpy, Batch
from datasets import facedataset
import config as cfg

CLASS_NAMES = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
MAX_IMAGES_PER_EXPRESSION = None


class AffectNet(facedataset.FaceDataset):

    classes = CLASS_NAMES
    colors = ['tab:gray', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:red', 'tab:blue']
    markers = ['s', 'o', '>', '<', '^', 'v', 'P', 'd']

    def __init__(self, root, cache_root=None, crop_source='bb_ground_truth', **kwargs):
        assert(crop_source in ['bb_ground_truth', 'lm_ground_truth', 'lm_cnn', 'lm_openface'])

        fullsize_img_dir = os.path.join(root, 'cropped_Annotated')
        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=fullsize_img_dir,
                         crop_source=crop_source,
                         crop_border_mode='mirror',
                         **kwargs)

        self.rebalance_classes()


    def _load_annotations(self, split):
        annotation_filename = 'training' if self.train else 'validation'
        path_annotations_mod = os.path.join(self.root, annotation_filename + '.mod.pkl')
        if os.path.isfile(path_annotations_mod):
            annotations = pd.read_pickle(path_annotations_mod)
        else:
            print('Reading CSV file...')
            annotations = pd.read_csv(os.path.join(self.root, annotation_filename+'.csv'))
            print('done.')

            # drop non-faces
            annotations = annotations[annotations.expression < 8]

            # Samples in annotation file are somewhat clustered by expression.
            # Shuffle to create a more even distribution.
            # NOTE: deterministic, always creates the same order
            if self.train:
                from sklearn.utils import shuffle
                annotations = shuffle(annotations, random_state=2)
                # remove samples with inconsistent expression<->valence/arousal values
                # self._remove_outliers()

            # self.annotations.to_csv(path_annotations_mod, index=False)
            annotations.to_pickle(path_annotations_mod)

        # There is (at least) one missing image in the dataset. Remove by checking face width:
        annotations = annotations[annotations.face_width > 0]
        return annotations


    def filter_labels(self, label_dict=None, label_dict_exclude=None):
        if label_dict is not None:
            print("Applying include filter to labels: {}".format(label_dict))
            for k, v in label_dict.items():
                self.annotations = self.annotations[self.annotations[k] == v]
        if label_dict_exclude is not None:
            print("Applying exclude filter to labels: {}".format(label_dict_exclude))
            for k, v in label_dict_exclude.items():
                self.annotations = self.annotations[self.annotations[k] != v]
        print("  Number of images: {}".format(len(self.annotations)))


    def rebalance_classes(self, max_images_per_class=MAX_IMAGES_PER_EXPRESSION):
        if max_images_per_class is not None and self.train:
            self._load_annotations(self.split)
            # balance class sized if neccessary
            print('Limiting number of images to {} per class...'.format(max_images_per_class))
            # self._annotations = self._annotations.groupby('expression').head(5000)
            from sklearn.utils import shuffle
            self.annotations['cls_idx'] = self.annotations.groupby('expression').cumcount()
            self.annotations = shuffle(self.annotations)
            self.annotations_balanced = self.annotations[self.annotations.cls_idx < max_images_per_class]
            self._select_index_range()

    @property
    def labels(self):
        return self.annotations['expression'].values

    @property
    def heights(self):
        return self.annotations.face_height.values

    @property
    def widths(self):
        return self.annotations.face_width.values

    def extra_repr(self):
        labels = self.annotations.expression
        fmt_str =  "    Class sizes:\n"
        for id in np.unique(labels):
            count = len(np.where(labels == id)[0])
            fmt_str += "      {:<6} ({:.2f}%)\t({})\n".format(count, 100.0*count/self.__len__(), self.classes[id])
        fmt_str += "    --------------------------------\n"
        fmt_str += "      {:<6}\n".format(len(labels))
        return fmt_str

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def _get_expression(sample):
        return np.array([[sample.expression, sample.valence, sample.arousal]], dtype=np.float32)

    def get_class_sizes(self):
        groups = self.annotations.groupby(by='expression')
        return groups.size().values

    def parse_landmarks(self, landmarks):
        try:
            vals = [float(s) for s in landmarks.split(';')]
            return np.array([(x, y) for x, y in zip(vals[::2], vals[1::2])], dtype=np.float32)
        except:
            raise ValueError("Invalid landmarks {}".format(landmarks))

    def get_crop_extend_factors(self):
        return 0.05, 0.25

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        filename = sample.subDirectory_filePath
        bb = [sample.face_x, sample.face_y, sample.face_width, sample.face_height]
        bb = geometry.extend_bbox(bb, dt=0.05, db=0.25)
        landmarks_to_return = self.parse_landmarks(sample.facial_landmarks)
        landmarks_for_crop = landmarks_to_return if self.crop_source == 'lm_ground_truth' else None
        return self.get_sample(filename, bb, landmarks_for_crop, landmarks_to_return=landmarks_to_return)


cfg.register_dataset(AffectNet)


if __name__ == '__main__':
    import argparse
    import torch
    from csl_common.vis import vis
    import config

    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', default=False, type=bool)
    parser.add_argument('--st', default=None, type=int)
    parser.add_argument('--nd', default=None, type=int)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dirs = config.get_dataset_paths('affectnet')
    train = True
    ds = AffectNet(root=dirs[0], image_size=256, cache_root=dirs[1], train=train, use_cache=False,
                   transform=ds_utils.build_transform(deterministic=not train, daug=0),
                   crop_source='lm_ground_truth')
    dl = td.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
    # print(ds)

    for data in dl:
        batch = Batch(data, gpu=False)

        gt = to_numpy(batch.landmarks)
        ocular_dists_inner = np.sqrt(np.sum((gt[:, 42] - gt[:, 39]) ** 2, axis=1))
        ocular_dists_outer = np.sqrt(np.sum((gt[:, 45] - gt[:, 36]) ** 2, axis=1))
        ocular_dists = np.vstack((ocular_dists_inner, ocular_dists_outer)).mean(axis=0)
        print(ocular_dists)

        images = vis.to_disp_images(batch.images, denorm=True)
        imgs = vis.add_landmarks_to_images(images, batch.landmarks.numpy())
        vis.vis_square(imgs, nCols=10, fx=1.0, fy=1.0, normalize=False)