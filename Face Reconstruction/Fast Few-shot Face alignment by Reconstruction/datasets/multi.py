import numpy as np
import torch.utils.data as td
import sklearn.utils


class ConcatDataset(td.Dataset):
    def __init__(self, datasets, shuffle=True, max_samples=None):
        self.datasets = datasets

        inds = []
        for id, ds in enumerate(self.datasets):
            inds.append(np.array([[id]*len(ds), range(len(ds))]))

        self.joint_idx = np.hstack(inds).transpose()
        if shuffle:
            self.joint_idx = sklearn.utils.shuffle(self.joint_idx)

        if max_samples is not None:
            self.joint_idx = self.joint_idx[:max_samples]

    def __len__(self):
        return len(self.joint_idx)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.joint_idx[idx]
        return self.datasets[ds_idx][sample_idx]

    def __repr__(self):
        return '\n'.join([ds.__repr__() for ds in self.datasets])

    def print_stats(self):
        for ds in self.datasets:
            ds.print_stats()

    def get_class_sizes(self):
        sizes = []
        for ds in self.datasets:
            sizes.append(ds.get_class_sizes())
        return np.sum(sizes, axis=0)


class ConcatFaceDataset(ConcatDataset):

    @property
    def heights(self):
        sizes = []
        for ds in self.datasets:
            sizes.append(ds.heights)
        return np.concatenate(sizes)

    @property
    def widths(self):
        sizes = []
        for ds in self.datasets:
            sizes.append(ds.widths)
        return np.concatenate(sizes)


