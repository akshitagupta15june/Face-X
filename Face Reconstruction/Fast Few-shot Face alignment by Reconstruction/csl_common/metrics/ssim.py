import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def structsim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    # padd = 10
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma1 = sigma1_sq.clamp(min=0)**0.5
    sigma2 = sigma2_sq.clamp(min=0)**0.5
    # sigma1_test = F.conv2d((img1 - mu1)**2, window, padding=padd, groups=channel)
    # sigma1_sq_test = sigma1_test.pow(2)


    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # v1 = 2.0 * sigma12 + C2
    # v2 = sigma1_sq + sigma2_sq + C2
    # cs = torch.mean(v1 / v2)  # contrast sensitivity

    l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    c = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    s = (sigma12 + (C2/2.0)) / (sigma1 * sigma2  + (C2/2.0))

    # cs_mean = (c * s).mean()
    # cs_mean_prod = c.mean() * s.mean()

    return l.mean(), c.mean(), s.mean()


# from datasets.ds_utils import denormalized
def ssim(img1, img2, window_size=-1, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    # img1 = denormalized(img1).clamp(min=0)
    # img2 = denormalized(img2).clamp(min=0)

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        if window_size > 0:
            real_size = min(window_size, height, width)
            window = create_window(real_size, channel=channel).to(img1.device)
        else:
            raise ValueError("If window not supplied, window_size must be non negative.")

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # sigma1_sq = sigma1_sq.clamp(min=0)**0.5
    # sigma2_sq = sigma2_sq.clamp(min=0)**0.5
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs_map = v1 / v2  # contrast sensitivity
    cs = torch.mean(cs_map)  # contrast sensitivity

    # sigma1 = sigma1_sq.clamp(min=0)**0.5
    # sigma2 = sigma2_sq.clamp(min=0)**0.5
    # c = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)

    # ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if full:
        return cs, cs_map
    return cs



def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False, window=None):
    device = img1.device
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001]).to(device)
    levels = weights.size()[0]
    # mssim = []
    mcs = []
    for _ in range(levels):
        cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=False, val_range=val_range, window=window)
        # mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    # mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        # mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    # pow1 = mcs ** weights
    # pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    # output = torch.prod(pow1[:-1]) * pow2[-1]
    output = mcs[1]
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
        self.cs_map = None

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        cs, self.cs_map =  ssim(img1, img2, -1, window, size_average=self.size_average, full=True)
        return cs


def msssim_simple(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False, window=None):
    k = (2,2)
    img1 = F.avg_pool2d(img1, k)
    img2 = F.avg_pool2d(img2, k)
    cs = ssim(img1, img2, window_size=window_size, size_average=size_average, val_range=val_range, window=window)
    return cs


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = None
        self.window = None

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if self.window is not None and channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            print("MSSIM: Creating window...")
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, window=window)
        return msssim_simple(img1, img2, window_size=self.window_size, size_average=self.size_average, window=window)
