import cv2
import numpy as np
import landmarks.lmconfig as lmcfg
from csl_common.utils.nn import to_numpy, to_image
from csl_common.vis import vis
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import torch
import warnings

layers = []

outline = range(0, 17)
eyebrow_l = range(17, 22)
eyebrow_r = range(22, 27)
nose = range(27, 31)
nostrils = range(31, 36)
eye_l = range(36, 42)
eye_r = range(42, 48)
mouth = range(48, 68)

components = [outline, eyebrow_l, eyebrow_r, nose, nostrils, eye_l, eye_r, mouth]

new_layers = []
for idx in range(20):
    lm_ids = []
    for comp in components[1:]:
        if len(comp) > idx:
            lm = comp[idx]
            lm_ids.append(lm)
    new_layers.append(lm_ids)

outline_layers = [[lm] for lm in range(17)]

layers = components + new_layers + outline_layers

hm_code_mat = np.zeros((len(layers), 68), dtype=bool)
for l, lm_ids in enumerate(layers):
    hm_code_mat[l, lm_ids] = True


def generate_colors(n, r, g, b, dim):
    ret = []
    step = [0,0,0]
    step[dim] =  256 / n
    for i in range(n):
        r += step[0]
        g += step[1]
        b += step[2]
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

_colors = generate_colors(17, 220, 0, 0, 2) + \
          generate_colors(10, 0, 240, 0, 0) + \
          generate_colors(9, 0, 0, 230, 1) + \
          generate_colors(12, 100, 255, 0, 2) + \
          generate_colors(20, 150, 0, 255, 2)
# lmcolors = np.array(_colors)
np.random.seed(0)
lmcolors = np.random.randint(0,255,size=(68,3))
lmcolors = lmcolors / lmcolors.sum(axis=1).reshape(-1,1)*255


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x-mean)**2 / (2 * sigma**2))


def make_landmark_template(wnd_size, sigma):
    X, Y = np.mgrid[-wnd_size//2:wnd_size//2, -wnd_size//2:wnd_size//2]
    Z = np.sqrt(X**2 + Y**2)
    N = gaussian(Z, 0, sigma)
    # return (N/N.max())**2  # square to make sharper
    return N / N.max()


def _fill_heatmap_layer(dst, lms, lm_id, lm_heatmap_window, wnd_size):
    posx, posy = min(lms[lm_id,0], lmcfg.HEATMAP_SIZE-1), min(lms[lm_id,1], lmcfg.HEATMAP_SIZE-1)

    img_size = lmcfg.HEATMAP_SIZE
    l = int(posx - wnd_size/2)
    t = int(posy - wnd_size/2)
    r = l + wnd_size
    b = t + wnd_size

    src_l = max(0, -l)
    src_t = max(0, -t)
    src_r = min(wnd_size, wnd_size-(r-img_size))
    src_b = min(wnd_size, wnd_size-(b-img_size))

    try:
        cn = lm_id
        wnd = lm_heatmap_window[src_t:src_b, src_l:src_r]
        weight = 1.0
        dst[cn, max(0,t):min(img_size, b), max(0,l):min(img_size, r)] = np.maximum(
            dst[cn, max(0,t):min(img_size, b), max(0,l):min(img_size, r)], wnd*weight)
    except:
        pass


def __get_code_mat(num_landmarks):
    def to_binary(n, ndigits):
        bits =  np.array([bool(int(x)) for x in bin(n)[2:]])
        assert len(bits) <= ndigits
        zero_pad_bits = np.zeros(ndigits, dtype=bool)
        zero_pad_bits[-len(bits):] = bits
        return zero_pad_bits

    n_enc_layers = int(np.ceil(np.log2(num_landmarks)))

    # get binary code for each heatmap id
    codes = [to_binary(i+1, ndigits=n_enc_layers) for i in range(num_landmarks)]
    return np.vstack(codes)


def convert_to_encoded_heatmaps(hms):

    def merge_layers(hms):
        hms = hms.max(axis=0)
        return hms/hms.max()

    num_landmarks = len(hms)
    n_enc_layers = len(hm_code_mat)

    # create compressed heatmaps by merging layers according to transpose of binary code mat
    encoded_hms = np.zeros((n_enc_layers, hms.shape[1], hms.shape[2]))
    for l in range(n_enc_layers):
        selected_layer_ids = hm_code_mat[l,:]
        encoded_hms[l] = merge_layers(hms[selected_layer_ids].copy())
    decode_heatmaps(encoded_hms)
    return encoded_hms


def convert_to_hamming_encoded_heatmaps(hms):

    def merge_layers(hms):
        hms = hms.max(axis=0)
        return hms/hms.max()

    num_landmarks = len(hms)
    n_enc_layers = int(np.ceil(np.log2(num_landmarks)))
    code_mat = __get_code_mat(num_landmarks)

    # create compressed heatmaps by merging layers according to transpose of binary code mat
    encoded_hms = np.zeros((n_enc_layers, hms.shape[1], hms.shape[2]))
    for l in range(n_enc_layers):
        selected_layer_ids = code_mat[:, l]
        encoded_hms[l] = merge_layers(hms[selected_layer_ids].copy())
    # decode_heatmaps(encoded_hms)
    return encoded_hms


def decode_heatmaps(hms):
    import cv2
    def get_decoded_heatmaps_for_layer(hms, lm):
        show = False
        enc_layer_ids = code_mat[:, lm]
        heatmap = np.ones_like(hms[0])
        for i in range(len(enc_layer_ids)):
            pos = enc_layer_ids[i]
            layer = hms[i]
            if pos:
                if show:
                    fig, ax = plt.subplots(1,4)
                    print(i, pos)
                    ax[0].imshow(heatmap, vmin=0, vmax=1)
                    ax[1].imshow(layer, vmin=0, vmax=1)
                # mask = layer.copy()
                # mask[mask < 0.1] = 0
                # heatmap *= mask
                heatmap *= layer
                if show:
                    # ax[2].imshow(mask, vmin=0, vmax=1)
                    ax[3].imshow(heatmap, vmin=0, vmax=1)

        return heatmap

    num_landmarks = 68

    # get binary code for each heatmap id
    code_mat = hm_code_mat

    decoded_hms = np.zeros((num_landmarks, hms.shape[1], hms.shape[1]))

    show = False
    if show:
        fig, ax = plt.subplots(1)
        ax.imshow(code_mat)
        fig_dec, ax_dec = plt.subplots(7, 10)
        fig, ax = plt.subplots(5,9)
        for i in range(len(hms)):
            ax[i//9, i%9].imshow(hms[i])

    lms = np.zeros((68,2), dtype=int)
    # lmid_to_show = 16

    for lm in range(0,68):

        heatmap = get_decoded_heatmaps_for_layer(hms, lm)

        decoded_hms[lm] = heatmap
        heatmap = cv2.blur(heatmap, (5, 5))
        lms[lm, :] = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[::-1]

        if show:
            ax_dec[lm//10, lm%10].imshow(heatmap)

    if show:
        plt.show()

    return decoded_hms, lms


def create_landmark_heatmaps(lms, sigma, landmarks_to_use, heatmap_size):
    lm_wnd_size = int(sigma * 5)
    lm_heatmap_window = make_landmark_template(lm_wnd_size, sigma)
    nchannels = len(landmarks_to_use)

    hms = np.zeros((nchannels, heatmap_size, heatmap_size))
    for l in landmarks_to_use:
        wnd = lm_heatmap_window
        _fill_heatmap_layer(hms, lms, l, wnd, lm_wnd_size)

    if lmcfg.LANDMARK_TARGET == 'single_channel':
        hms = hms.max(axis=0)
        hms /= hms.max()
    return hms.astype(np.float32)


def calc_landmark_nme_per_img(gt_lms, pred_lms, ocular_norm='pupil', landmarks_to_eval=None, image_size=None):
    norm_dists = calc_landmark_nme(gt_lms, pred_lms, ocular_norm, image_size=image_size)
    if landmarks_to_eval is None:
        landmarks_to_eval = range(norm_dists.shape[1])
    return np.mean(np.array([norm_dists[:,l] for l in landmarks_to_eval]).T, axis=1)


def get_pupil_dists(lms):
    assert lms.shape[2] == 68
    ocular_dists_inner = np.sqrt(np.sum((lms[:, 42] - lms[:, 39])**2, axis=1))
    ocular_dists_outer = np.sqrt(np.sum((lms[:, 45] - lms[:, 36])**2, axis=1))
    return np.vstack((ocular_dists_inner, ocular_dists_outer)).mean(axis=0)


def get_landmark_confs(X_lm_hm):
    return np.clip(to_numpy(X_lm_hm).reshape(X_lm_hm.shape[0], X_lm_hm.shape[1], -1).max(axis=2), a_min=0, a_max=1)


def calc_landmark_nme(gt_lms, pred_lms, ocular_norm='pupil', image_size=None):
    def reformat(lms):
        lms = to_numpy(lms)
        if len(lms.shape) == 2:
            lms = lms.reshape((1,-1,2))
        return lms
    gt = reformat(gt_lms)
    pred = reformat(pred_lms)
    assert(len(gt.shape) == 3)
    assert(len(pred.shape) == 3)
    if gt.shape[1] == 5:  # VGGFace2
       ocular_dists = np.sqrt(np.sum((gt[:, 1] - gt[:, 0])**2, axis=1))
    elif gt.shape[1] == 19:  # AFLW
        assert image_size is not None
        ocular_dists = np.ones(gt.shape[0], dtype=np.float32) * image_size
    elif gt.shape[1] == 98:  # WFLW
        ocular_dists = np.sqrt(np.sum((gt[:, 72] - gt[:, 60])**2, axis=1))
    elif gt.shape[1] == 68:  # 300-W
        if ocular_norm == 'pupil':
            ocular_dists = get_pupil_dists(gt)
        elif ocular_norm == 'outer':
            ocular_dists = np.sqrt(np.sum((gt[:, 45] - gt[:, 36])**2, axis=1))
        elif ocular_norm is None or ocular_norm == 'none':
            ocular_dists = np.ones((len(gt),1)) * 100.0
        else:
            raise ValueError("Ocular norm {} not defined!".format(ocular_norm))
    else:
        assert image_size is not None
        ocular_dists = np.ones(gt.shape[0], dtype=np.float32) * image_size
    norm_dists = np.sqrt(np.sum((gt - pred)**2, axis=2)) / ocular_dists.reshape(len(gt), 1)
    return norm_dists * 100


def calc_landmark_failure_rate(nmes, th=10.0):
    img_nmes = nmes.mean(axis=1)
    assert len(img_nmes) == len(nmes)
    return np.count_nonzero(img_nmes > th) / len(img_nmes.ravel())


def calc_landmark_ncc(X, X_recon, lms):
    input_images = vis.to_disp_images(X, denorm=True)
    recon_images = vis.to_disp_images(X_recon, denorm=True)
    nimgs = len(input_images)
    nlms = len(lms[0])
    wnd_size = (X_recon.shape[-1] // 16) - 1
    nccs = np.zeros((nimgs, nlms), dtype=np.float32)
    img_shape = input_images[0].shape
    for i in range(nimgs):
        for lid in range(nlms):
            x = int(lms[i, lid, 0])
            y = int(lms[i, lid, 1])
            t = max(0, y-wnd_size//2)
            b = min(img_shape[0]-1, y+wnd_size//2)
            l = max(0, x-wnd_size//2)
            r = min(img_shape[1]-1, x+wnd_size//2)
            wnd1 = input_images[i][t:b, l:r]
            wnd2 = recon_images[i][t:b, l:r]
            ncc = ((wnd1-wnd1.mean()) * (wnd2-wnd2.mean())).mean() / (wnd1.std() * wnd2.std())
            nccs[i, lid] = ncc
    return np.clip(np.nan_to_num(nccs), a_min=-1, a_max=1)


def calc_landmark_ssim_score(X, X_recon, lms, wnd_size=None):
    if wnd_size is None:
        wnd_size = (X_recon.shape[-1] // 16) - 1
    input_images = vis.to_disp_images(X, denorm=True)
    recon_images = vis.to_disp_images(X_recon, denorm=True)
    data_range = 255.0 if input_images[0].dtype == np.uint8 else 1.0
    nimgs = len(input_images)
    nlms = len(lms[0])
    scores = np.zeros((nimgs, nlms), dtype=np.float32)
    for i in range(nimgs):
        S = compare_ssim(input_images[i], recon_images[i], data_range=data_range, multichannel=True, full=True)[1]
        S = S.mean(axis=2)
        for lid in range(nlms):
            x = int(lms[i, lid, 0])
            y = int(lms[i, lid, 1])
            t = max(0, y-wnd_size//2)
            b = min(S.shape[0]-1, y+wnd_size//2)
            l = max(0, x-wnd_size//2)
            r = min(S.shape[1]-1, x+wnd_size//2)
            wnd = S[t:b, l:r]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scores[i, lid] = wnd.mean()
    return np.nan_to_num(scores)


def calc_landmark_cs_error(X, X_recon, lms, torch_ssim, training=False, wnd_size=None):
    if wnd_size is None:
        wnd_size = (X_recon.shape[-1] // 16) - 1
    nimgs = len(X)
    nlms = len(lms[0])
    input_size = X.shape[-1]
    errs = torch.zeros((nimgs, nlms), requires_grad=training).cuda()
    for i in range(len(X)):
        torch_ssim(X[i].unsqueeze(0), X_recon[i].unsqueeze(0))
        cs_map = torch_ssim.cs_map[0].mean(dim=0)
        map_size = cs_map.shape[0]
        margin = (input_size - map_size) // 2
        S = torch.zeros((input_size, input_size), requires_grad=training).cuda()
        S[margin:-margin, margin:-margin] = cs_map
        for lid in range(nlms):
            x = int(lms[i, lid, 0])
            y = int(lms[i, lid, 1])
            t = max(0, y-wnd_size//2)
            b = min(S.shape[0]-1, y+wnd_size//2)
            l = max(0, x-wnd_size//2)
            r = min(S.shape[1]-1, x+wnd_size//2)
            wnd = S[t:b, l:r]
            errs[i, lid] = 1 - wnd.mean()
    return errs


def calc_landmark_recon_error(X, X_recon, lms, return_maps=False, reduction='mean'):
    assert len(X.shape) == 4
    assert reduction in ['mean', 'none']
    X = to_numpy(X)
    X_recon = to_numpy(X_recon)
    mask = np.zeros((X.shape[0], X.shape[2], X.shape[3]), dtype=np.float32)
    input_size = X.shape[-1]
    radius = input_size * 0.05
    for img_id in range(len(mask)):
        for lm in lms[img_id]:
            cv2.circle(mask[img_id], (int(lm[0]), int(lm[1])), radius=int(radius), color=1, thickness=-1)
    err_maps = np.abs(X - X_recon).mean(axis=1) * 255.0
    masked_err_maps = err_maps * mask

    debug = False
    if debug:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(vis.to_disp_image((X * mask[:,np.newaxis,:,:].repeat(3, axis=1))[0], denorm=True))
        ax[1].imshow(vis.to_disp_image((X_recon * mask[:,np.newaxis,:,:].repeat(3, axis=1))[0], denorm=True))
        ax[2].imshow(masked_err_maps[0])
        plt.show()

    if reduction == 'mean':
        err = masked_err_maps.sum() / (mask.sum() * 3)
    else:
        # err = masked_err_maps.mean(axis=2).mean(axis=1)
        err = masked_err_maps.sum(axis=2).sum(axis=1) / (mask.reshape(len(mask), -1).sum(axis=1) * 3)

    if return_maps:
        return err, masked_err_maps
    else:
        return err


def to_single_channel_heatmap(lm_heatmaps):
    if lmcfg.LANDMARK_TARGET == 'colored':
        mc = [to_image(lm_heatmaps[0])]
    elif lmcfg.LANDMARK_TARGET == 'single_channel':
        mc = [to_image(lm_heatmaps[0, 0])]
    else:
        mc = to_image(lm_heatmaps.max(axis=1))
    return mc


LM68_TO_LM96 = 1
LM98_TO_LM68 = 2
LM68_TO_LM5 = 3

def convert_landmarks(lms, code):
    cvt_func = {
        LM98_TO_LM68: lm98_to_lm68,
        LM68_TO_LM5: lm68_to_lm5,
    }
    if len(lms.shape) == 3:
        # new_lms = []
        new_lms = torch.zeros((len(lms), 5, 2), requires_grad=True, dtype=torch.float32).cuda()
        for i in range(len(lms)):
            # new_lms.append(cvt_func[code](lms[i]))
            # new_lms[i] += cvt_func[code](lms[i])
            cvt_func[code](lms[i], new_lms[i])
        # new_lms = torch.stack(new_lms)
        # new_lms = np.array(new_lms)
        return new_lms
    elif len(lms.shape) == 2:
        return cvt_func[code](lms)
    else:
        raise ValueError


def lm68_to_lm5(lm68, lm5):
    # lm5 = np.zeros((len(lm68), 5, 2), dtype=np.float32)
    # lm5 = torch.zeros((5, 2), requires_grad=True, dtype=torch.float32)
    lm5[0] += (lm68[36] + lm68[39]) / 2
    lm5[1] += (lm68[42] + lm68[45]) / 2
    lm5[2] += lm68[30]
    lm5[3] += lm68[48]
    lm5[4] += lm68[54]
    # return lm5


def lm98_to_lm68(lm98):
    def copy_lms(offset68, offset98, n):
        lm68[range(offset68, offset68+n)] = lm98[range(offset98, offset98+n)]

    assert len(lm98) == 98, "Cannot convert landmarks to 68 points!"
    lm68 = np.zeros((68,2), dtype=np.float32)

    # outline
    lm68[range(17)] = lm98[range(0,33,2)]

    # left eyebrow
    copy_lms(17, 33, 5)
    # right eyebrow
    copy_lms(22, 42, 5)
    # nose
    copy_lms(27, 51, 9)

    # eye left
    lm68[36] = lm98[60]
    lm68[37] = lm98[61]
    lm68[38] = lm98[63]
    lm68[39] = lm98[64]
    lm68[40] = lm98[65]
    lm68[41] = lm98[67]

    # eye right
    lm68[36+6] = lm98[60+8]
    lm68[37+6] = lm98[61+8]
    lm68[38+6] = lm98[63+8]
    lm68[39+6] = lm98[64+8]
    lm68[40+6] = lm98[65+8]
    lm68[41+6] = lm98[67+8]

    copy_lms(48, 76, 20)

    return lm68


def is_good_landmark(confs, rec_errs=None):
    if rec_errs is not None:
        low_errors = rec_errs < 0.25
        confs *= np.array(low_errors).astype(int)
    return confs > 0.8


def smooth_heatmaps(hms):
    assert(len(hms.shape) == 4)
    hms = to_numpy(hms)
    for i in range(hms.shape[0]):
        for l in range(hms.shape[1]):
            hms[i,l] = cv2.blur(hms[i,l], (9,9), borderType=cv2.BORDER_CONSTANT)
            # hms[i,l] = cv2.GaussianBlur(hms[i,l], (9,9), sigmaX=9, borderType=cv2.BORDER_CONSTANT)
    return hms


def heatmaps_to_landmarks(hms, target_size):
    def landmark_id_to_heatmap_id(lm_id):
        return {lm: i for i,lm in enumerate(range(num_landmarks))}[lm_id]

    assert len(hms.shape) == 4
    num_images = hms.shape[0]
    num_landmarks = hms.shape[1]
    heatmap_size = hms.shape[-1]
    lms = np.zeros((num_images, num_landmarks, 2), dtype=int)
    if hms.shape[1] > 3:
        # print(hms.max())
        for i in range(len(hms)):
            heatmaps = to_numpy(hms[i])
            for l in range(len(heatmaps)):
                hm = heatmaps[landmark_id_to_heatmap_id(l)]
                lms[i, l, :] = np.unravel_index(np.argmax(hm, axis=None), hm.shape)[::-1]
    lm_scale = heatmap_size / target_size
    return lms / lm_scale
