import json
import os

import numpy as np
import torch


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def to_numpy(ft):
    if isinstance(ft, np.ndarray):
        return ft
    try:
        return ft.detach().cpu().numpy()
    except AttributeError:
        return None


def to_image(m):
    img = to_numpy(m)
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    return img


def unsqueeze(x):
    if isinstance(x, np.ndarray):
        return x[np.newaxis, ...]
    else:
        return x.unsqueeze(dim=0)


def atleast4d(x):
    if len(x.shape) == 3:
        return unsqueeze(x)
    return x


def atleast3d(x):
    if len(x.shape) == 2:
        return unsqueeze(x)
    return x


class Batch:

    def __init__(self, data, n=None, gpu=True, eval=False):
        self.images = atleast4d(data['image'])

        self.eval = eval

        try:
            self.ids = data['id']
            try:
                if self.ids.min() < 0 or self.ids.max() == 0:
                    self.ids = None
            except AttributeError:
                self.ids = np.array(self.ids)
        except KeyError:
            self.ids = None

        try:
            self.target_images = data['target']
        except KeyError:
            self.target_images = None

        try:
            self.face_heights = data['face_heights']
        except KeyError:
            self.face_heights = None

        try:
            self.poses = data['pose']
        except KeyError:
            self.poses = None

        try:
            self.landmarks = atleast3d(data['landmarks'])
        except KeyError:
            self.landmarks = None

        try:
            self.clips = np.array(data['vid'])
        except KeyError:
            self.clips = None

        try:
            self.fnames = data['fnames']
        except:
            self.fnames = None

        try:
            self.bumps = data['bumps']
        except:
            self.bumps = None

        try:
            self.affine = data['affine']
        except:
            self.affine = None

        try:
            self.face_masks = data['face_mask']
        except:
            self.face_masks = None

        if self.face_masks is not None:
            self.face_weights = self.face_masks.float()
            if not self.eval:
                self.face_weights += 1.0
            self.face_weights /= self.face_weights.max()
            # plt.imshow(self.face_weights[0,0])
            # plt.show()

            # if cfg.WITH_FACE_MASK:
            #     mask = self.face_masks.unsqueeze(1).expand_as(self.images).float()
            #     mask /= mask.max()
            #     self.images *= mask

        self.lm_heatmaps = None

        try:
            # self.face_weights = data['face_weights']
            self.lm_heatmaps = data['lm_heatmaps']
            if len(self.lm_heatmaps.shape) == 3:
                self.lm_heatmaps = self.lm_heatmaps.unsqueeze(1)
        except KeyError:
            self.face_weights = 1.0

        for k, v in self.__dict__.items():
            if v is not None:
                try:
                    self.__dict__[k] = v[:n]
                except TypeError:
                    pass

        if gpu:
            for k, v in self.__dict__.items():
                if v is not None:
                    try:
                        self.__dict__[k] = v.cuda()
                    except AttributeError:
                        pass

    def __len__(self):
        return len(self.images)


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def read_model(in_dir, model_name, model):
    filepath_mdl = os.path.join(in_dir, model_name+'.mdl')
    snapshot = torch.load(filepath_mdl)
    try:
        model.load_state_dict(snapshot['state_dict'], strict=False)
    except RuntimeError as e:
        print(e)


def read_meta(in_dir):
    with open(os.path.join(in_dir, 'meta.json'), 'r') as outfile:
        data = json.load(outfile)
    return data


def denormalize(tensor):
    # assert(len(tensor.shape[1] == 3)
    if tensor.shape[1] == 3:
        tensor[:, 0] += 0.518
        tensor[:, 1] += 0.418
        tensor[:, 2] += 0.361
    elif tensor.shape[-1] == 3:
        tensor[..., 0] += 0.518
        tensor[..., 1] += 0.418
        tensor[..., 2] += 0.361


def denormalized(tensor):
    # assert(len(tensor.shape[1] == 3)
    if isinstance(tensor, np.ndarray):
        t = tensor.copy()
    else:
        t = tensor.clone()
    denormalize(t)
    return t