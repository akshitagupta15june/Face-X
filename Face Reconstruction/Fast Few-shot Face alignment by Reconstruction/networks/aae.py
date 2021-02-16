import cv2
import os
import numpy as np

import torch
import torch.nn as nn

import config as cfg
import networks.invresnet
import face_vis
from networks.archs import D_net_gauss, Discriminator
from networks import resnet_ae, archs

from csl_common.utils.nn import to_numpy, count_parameters, read_model, read_meta


def calc_acc(outputs, labels):
    assert(outputs.shape[1] == 8)
    assert(len(outputs) == len(labels))
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels)
    acc = corrects.double()/float(outputs.size(0))
    return acc.item()

def pearson_dist(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    r = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)))
    return 1 - r.abs().mean()


def resize_image_batch(X, target_size):
    resize = lambda im: cv2.resize(im, dsize=tuple(target_size), interpolation=cv2.INTER_CUBIC)
    X = X.cpu()
    imgs = [i.permute(1, 2, 0).numpy() for i in X]
    imgs = [resize(i) for i in imgs]
    tensors = [torch.from_numpy(i).permute(2, 0, 1) for i in imgs]
    return torch.stack(tensors)


def load_net(modelname):
    modelfile = os.path.join(cfg.SNAPSHOT_DIR, modelname)
    meta = read_meta(modelfile)
    input_size = meta.get('input_size', 256)
    output_size = meta.get('output_size', input_size)
    z_dim = meta.get('z_dim', 99)

    net = AAE(input_size=input_size, output_size=output_size, z_dim=z_dim)
    print("Loading model {}...".format(modelfile))
    read_model(modelfile, 'saae', net)
    print("Model trained for {} iterations.".format(meta['total_iter']))
    return net


class AAE(nn.Module):
    def __init__(self, input_size, output_size=None, pretrained_encoder=False,
                 z_dim=99):

        super(AAE, self).__init__()

        assert input_size in [128, 256, 512, 1024]

        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.z_dim = z_dim
        input_channels = 3

        self.Q = resnet_ae.resnet18(pretrained=pretrained_encoder,
                                    num_classes=self.z_dim,
                                    input_size=input_size,
                                    input_channels=input_channels,
                                    layer_normalization=cfg.ENCODER_LAYER_NORMALIZATION).cuda()

        decoder_class = networks.invresnet.InvResNet
        num_blocks = [cfg.DECODER_PLANES_PER_BLOCK] * 4
        self.P = decoder_class(networks.invresnet.InvBasicBlock,
                               num_blocks,
                               input_dims=self.z_dim,
                               output_size=output_size,
                               output_channels=input_channels,
                               layer_normalization=cfg.DECODER_LAYER_NORMALIZATION,
                               spectral_norm=cfg.DECODER_SPECTRAL_NORMALIZATION,
                               ).cuda()

        self.D_z = D_net_gauss(self.z_dim).cuda()
        self.D = Discriminator().cuda()

        print("Trainable params Q: {:,}".format(count_parameters(self.Q)))
        print("Trainable params P: {:,}".format(count_parameters(self.P)))
        print("Trainable params D_z: {:,}".format(count_parameters(self.D_z)))
        print("Trainable params D: {:,}".format(count_parameters(self.D)))

        self.total_iter = 0
        self.iter = 0
        self.z = None
        self.images = None
        self.current_dataset = None

    def z_vecs(self):
        return [to_numpy(self.z)]

    def forward(self, X):
        self.z = self.Q(X)
        outputs = self.P(self.z)
        self.landmark_heatmaps = None
        if outputs.shape[1] > 3:
            self.landmark_heatmaps = outputs[:,3:]
        return outputs[:,:3]


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

    return face_vis.draw_results(inputs_resized, X_recon, net.z_vecs(),
                                 landmarks=landmarks_resized,
                                 landmarks_pred=landmarks_pred,
                                 cs_errs=cs_errs,
                                 fx=fx, fy=fy, ncols=ncols)



