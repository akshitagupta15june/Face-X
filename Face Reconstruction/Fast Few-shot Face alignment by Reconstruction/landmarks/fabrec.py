import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as cfg
import networks.invresnet
from csl_common import vis

from csl_common.utils.nn import to_numpy, count_parameters
from csl_common import utils
import landmarks.lmconfig as lmcfg
import landmarks.lmutils
from networks import aae
from csl_common.utils import nn


def load_net(model, num_landmarks=None):
    meta = nn.read_meta(model)
    input_size = meta.get('input_size', 256)
    output_size = meta.get('output_size', input_size)
    if num_landmarks is None:
        num_landmarks = 98
    num_landmarks = meta.get('num_landmarks', num_landmarks)
    z_dim = meta.get('z_dim', 99)
    net = Fabrec(num_landarks=num_landmarks, input_size=input_size, output_size=output_size, z_dim=z_dim)
    print("Loading model {}...".format(model))
    nn.read_model(model, 'saae', net)
    return net



class Fabrec(aae.AAE):
    def __init__(self, num_landarks, **kwargs):
        super(Fabrec, self).__init__(**kwargs)

        self.num_landmark_heatmaps = num_landarks
        self.num_landmarks = num_landarks

        self.LMH = networks.invresnet.LandmarkHeadV2(networks.invresnet.InvBasicBlock,
                                                     [cfg.DECODER_PLANES_PER_BLOCK]*4,
                                                     output_size=lmcfg.HEATMAP_SIZE,
                                                     output_channels=self.num_landmark_heatmaps,
                                                     layer_normalization='batch').cuda()

        print("Trainable params LMH: {:,}".format(count_parameters(self.LMH)))

        self.total_iter = 0
        self.iter = 0
        self.z = None
        self.images = None
        self.current_dataset = None

    def z_vecs(self):
        return [to_numpy(self.z)]

    def heatmaps_to_landmarks(self, hms):
        lms = np.zeros((len(hms), self.num_landmarks, 2), dtype=int)
        if hms.shape[1] > 3:
            # print(hms.max())
            for i in range(len(hms)):
                heatmaps = to_numpy(hms[i])
                for l in range(len(heatmaps)):
                    hm = heatmaps[self.landmark_id_to_heatmap_id(l)]
                    lms[i, l, :] = np.unravel_index(np.argmax(hm, axis=None), hm.shape)[::-1]
        elif hms.shape[1] == 3:
            hms = to_numpy(hms)
            def get_score_plane(h, lm_id, cn):
                v = utils.nn.lmcolors[lm_id, cn]
                hcn = h[cn]
                hcn[hcn < v-2] = 0; hcn[hcn > v+5] = 0
                return hcn
            hms *= 255
            for i in range(len(hms)):
                hm = hms[i]
                for l in landmarks.config.LANDMARKS:
                    lm_score_map = get_score_plane(hm, l, 0) * get_score_plane(hm, l, 1) * get_score_plane(hm, l, 2)
                    lms[i, l, :] = np.unravel_index(np.argmax(lm_score_map, axis=None), lm_score_map.shape)[::-1]
        lm_scale = lmcfg.HEATMAP_SIZE / self.input_size
        return lms / lm_scale

    def landmarks_pred(self):
        try:
            if self.landmark_heatmaps is not None:
                return self.heatmaps_to_landmarks(self.landmark_heatmaps)
        except AttributeError:
            pass
        return None

    def detect_landmarks(self, X):
        X_recon = self.forward(X)
        X_lm_hm = self.LMH(self.P)
        X_lm_hm = landmarks.lmutils.smooth_heatmaps(X_lm_hm)
        lm_preds = to_numpy(self.heatmaps_to_landmarks(X_lm_hm))
        return X_recon, lm_preds, X_lm_hm

    def forward(self, X):
        self.z = self.Q(X)
        outputs = self.P(self.z)
        self.landmark_heatmaps = None
        if outputs.shape[1] > 3:
            self.landmark_heatmaps = outputs[:,3:]
        return outputs[:,:3]

    def landmark_id_to_heatmap_id(self, lm_id):
        return {lm: i for i,lm in enumerate(range(self.num_landmarks))}[lm_id]


def vis_reconstruction(net, inputs, landmarks=None, landmarks_pred=None,
                       pytorch_ssim=None, fx=0.5, fy=0.5, ncols=10):
    net.eval()
    cs_errs = None
    with torch.no_grad():
        X_recon = net(inputs)

        if pytorch_ssim is not None:
            cs_errs = np.zeros(len(inputs))
            for i in range(len(cs_errs)):
                cs_errs[i] = 1 - pytorch_ssim(inputs[i].unsqueeze(0), X_recon[i].unsqueeze(0)).item()

    inputs_resized = inputs
    landmarks_resized = landmarks
    if landmarks is not None:
        landmarks_resized = landmarks.cpu().numpy().copy()
        landmarks_resized[...,0] *= inputs_resized.shape[3]/inputs.shape[3]
        landmarks_resized[...,1] *= inputs_resized.shape[2]/inputs.shape[2]

    return vis.draw_results(inputs_resized, X_recon, net.z_vecs(),
                            landmarks=landmarks_resized,
                            landmarks_pred=landmarks_pred,
                            cs_errs=cs_errs,
                            fx=fx, fy=fy, ncols=ncols)



