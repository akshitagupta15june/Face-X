import cv2
from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
from csl_common.utils.nn import to_numpy


def denormalize(tensor):
    if tensor.shape[1] == 3:
        tensor[:, 0] += 0.518
        tensor[:, 1] += 0.418
        tensor[:, 2] += 0.361
    elif tensor.shape[-1] == 3:
        tensor[..., 0] += 0.518
        tensor[..., 1] += 0.418
        tensor[..., 2] += 0.361

def denormalized(tensor):
    if isinstance(tensor, np.ndarray):
        t = tensor.copy()
    else:
        t = tensor.clone()
    denormalize(t)
    return t

def color_map(data, vmin=None, vmax=None, cmap=plt.cm.viridis):
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    val = np.maximum(vmin, np.minimum(vmax, data))
    norm = (val-vmin)/(vmax-vmin)
    cm = cmap(norm)
    if isinstance(cm, tuple):
        return cm[:3]
    if len(cm.shape) > 2:
        cm = cm[:,:,:3]
    return cm


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def make_grid(data, padsize=2, padval=255, nCols=10, dsize=None, fx=None, fy=None, normalize=False):
    # if not isinstance(data, np.ndarray):
    data = np.array(data)
    if data.shape[0] == 0:
        return
    if data.shape[1] == 3:
        data = data.transpose((0,2,3,1))
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if normalize:
        data -= data.min()
        data /= data.max()
    else:
        data[data < 0] = 0
    #     data[data > 1] = 1

    # force the number of filters to be square
    # n = int(np.ceil(np.sqrt(data.shape[0])))
    c = nCols
    r = int(np.ceil(data.shape[0]/float(c)))

    padding = ((0, r*c - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((r, c) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((r * data.shape[1], c * data.shape[3]) + data.shape[4:])

    if dsize is not None or fx is not None or fy is not None:
        data = cv2.resize(data, dsize=dsize, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)

    return data


def vis_square(data, padsize=1, padval=0, wait=0, nCols=10, title='results', dsize=None, fx=None, fy=None, normalize=False):
    img = make_grid(data, padsize=padsize, padval=padval, nCols=nCols, dsize=dsize, fx=fx, fy=fy, normalize=normalize)
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)


def cvt32FtoU8(img):
    return (img * 255.0).astype(np.uint8)


def to_disp_image(img, denorm=False, output_dtype=np.uint8):
    if not isinstance(img, np.ndarray):
        img = img.detach().cpu().numpy()
    img = img.astype(np.float32).copy()
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    if denorm:
        img = denormalized(img)
    if img.max() > 2.00:
        if isinstance(img, np.ndarray):
            img /= 255.0
        else:
            raise ValueError("Image data in wrong value range (min/max={:.2f}/{:.2f}).".format(img.min(), img.max()))
    img = np.clip(img, a_min=0, a_max=1)
    if output_dtype == np.uint8:
        img = cvt32FtoU8(img)
    return img


def to_disp_images(images, denorm=False):
    return [to_disp_image(i, denorm) for i in images]


def add_frames_to_images(images, labels, label_colors, gt_labels=None):
    import collections
    if not isinstance(labels, (collections.Sequence, np.ndarray)):
        labels = [labels] * len(images)
    new_images = to_disp_images(images)
    for idx, (disp, label) in enumerate(zip(new_images, labels)):
        frame_width = 3
        bgr = label_colors[label]
        cv2.rectangle(disp,
                      (frame_width // 2, frame_width // 2),
                      (disp.shape[1] - frame_width // 2, disp.shape[0] - frame_width // 2),
                      color=bgr,
                      thickness=frame_width)

        if gt_labels is not None:
            radius = 8
            color = (0, 1, 0) if gt_labels[idx] == label else (1, 0, 0)
            cv2.circle(disp, (disp.shape[1] - 2*radius, 2*radius), radius, color, -1)
    return new_images


def add_cirle_to_images(images, intensities, cmap=plt.cm.viridis, radius=10):
    new_images = to_disp_images(images)
    for idx, (disp, val) in enumerate(zip(new_images, intensities)):
        # color = (0, 1, 0) if gt_labels[idx] == label else (1, 0, 0)
        # color = plt_colors.to_rgb(val)
        if isinstance(val, float):
            color = cmap(val).ravel()
        else:
            color = val
        cv2.circle(disp, (2*radius, 2*radius), radius, color, -1)
        # new_images.append(disp)
    return new_images


def get_pos_in_image(loc, text_size, image_shape):
    bottom_offset = int(6*text_size)
    right_offset = int(95*text_size)
    line_height = int(35*text_size)
    mid_offset = right_offset
    top_offset = line_height + int(0.05*line_height)
    if loc == 'tl':
        pos = (2, top_offset)
    elif loc == 'tr':
        pos = (image_shape[1]-right_offset, top_offset)
    elif loc == 'tr+1':
        pos = (image_shape[1]-right_offset, top_offset + line_height)
    elif loc == 'tr+2':
        pos = (image_shape[1]-right_offset, top_offset + line_height*2)
    elif loc == 'bl':
        pos = (2, image_shape[0]-bottom_offset)
    elif loc == 'bl-1':
        pos = (2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bl-2':
        pos = (2, image_shape[0]-bottom_offset-2*line_height)
    # elif loc == 'bm':
    #     pos = (mid_offset, image_shape[0]-bottom_offset)
    # elif loc == 'bm-1':
    #     pos = (mid_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset)
    elif loc == 'br-1':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br-2':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-2*line_height)
    elif loc == 'bm':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset)
    elif loc == 'bm-1':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bm-2':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset-2*line_height)
    else:
        raise ValueError("Unknown location {}".format(loc))
    return pos


def add_id_to_images(images, ids, gt_ids=None, loc='tl', color=(1,1,1), size=0.7, thickness=1):
    new_images = to_disp_images(images)
    for idx, (disp, val) in enumerate(zip(new_images, ids)):
        if gt_ids is not None:
            color = (0,1,0) if ids[idx] == gt_ids[idx] else (1,0,0)
        # if val != 0:
        pos = get_pos_in_image(loc, size, disp.shape)
        cv2.putText(disp, str(val), pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def add_error_to_images(images, errors, loc='bl', size=0.65, vmin=0., vmax=30.0, thickness=1,
                        format_string='{:.1f}', colors=None):
    new_images = to_disp_images(images)
    if colors is None:
        colors = color_map(to_numpy(errors), cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        if images[0].dtype == np.uint8:
            colors *= 255
    for disp, err, color in zip(new_images, errors, colors):
        pos = get_pos_in_image(loc, size, disp.shape)
        cv2.putText(disp, format_string.format(err), pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def add_landmarks_to_images(images, landmarks, color=None, radius=2, gt_landmarks=None,
                            lm_errs=None, lm_confs=None, lm_rec_errs=None,
                            draw_dots=True, draw_wireframe=False, draw_gt_offsets=False, landmarks_to_draw=None,
                            offset_line_color=None):

    def draw_wireframe_lines(img, lms):
        pts = lms.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(img, [pts[:17]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # head outline
        cv2.polylines(img, [pts[17:22]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # left eyebrow
        cv2.polylines(img, [pts[22:27]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # right eyebrow
        cv2.polylines(img, [pts[27:31]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # nose vert
        cv2.polylines(img, [pts[31:36]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # nose hor
        cv2.polylines(img, [pts[36:42]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # left eye
        cv2.polylines(img, [pts[42:48]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # right eye
        cv2.polylines(img, [pts[48:60]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # outer mouth
        cv2.polylines(img, [pts[60:68]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # inner mouth

    def draw_wireframe_lines_98(img, lms):
        pts = lms.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(img, [pts[:33]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # head outline
        cv2.polylines(img, [pts[33:42]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # left eyebrow
        # cv2.polylines(img, [pts[38:42]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # right eyebrow
        cv2.polylines(img, [pts[42:51]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # nose vert
        cv2.polylines(img, [pts[51:55]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # nose hor
        cv2.polylines(img, [pts[55:60]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # left eye
        cv2.polylines(img, [pts[60:68]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # right eye
        cv2.polylines(img, [pts[68:76]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # outer mouth
        cv2.polylines(img, [pts[76:88]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # inner mouth
        cv2.polylines(img, [pts[88:96]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # inner mouth

    def draw_offset_lines(img, lms, gt_lms, errs):
        if gt_lms.sum() == 0:
            return
        if lm_errs is None:
            # if offset_line_color is None:
            offset_line_color = (1,1,1)
            colors = [offset_line_color] * len(lms)
        else:
            colors = color_map(errs, cmap=plt.cm.jet, vmin=0, vmax=15.0)
        if img.dtype == np.uint8:
            colors *= 255
        for i, (p1, p2) in enumerate(zip(lms, gt_lms)):
            if landmarks_to_draw is None or i in landmarks_to_draw:
                if p1.min() > 0:
                    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), colors[i], thickness=1, lineType=cv2.LINE_AA)

    new_images = to_disp_images(images)
    landmarks = to_numpy(landmarks)
    gt_landmarks = to_numpy(gt_landmarks)
    lm_errs = to_numpy(lm_errs)
    img_size = new_images[0].shape[0]
    default_color = (255,255,255)

    if gt_landmarks is not None and draw_gt_offsets:
        for img_id  in range(len(new_images)):
            if gt_landmarks[img_id].sum() == 0:
                continue
            dists = None
            if lm_errs is not None:
                dists = lm_errs[img_id]
            draw_offset_lines(new_images[img_id], landmarks[img_id], gt_landmarks[img_id], dists)

    for img_id, (disp, lm)  in enumerate(zip(new_images, landmarks)):
        if len(lm) in [68, 21, 19, 98, 8, 5, 38]:
            if draw_dots:
                for lm_id in range(0,len(lm)):
                    if landmarks_to_draw is None or lm_id in landmarks_to_draw or len(lm) != 68:
                        lm_color = color
                        if lm_color is None:
                            if lm_errs is not None:
                                lm_color = color_map(lm_errs[img_id, lm_id], cmap=plt.cm.jet, vmin=0, vmax=1.0)
                            else:
                                lm_color = default_color
                        # if lm_errs is not None and lm_errs[img_id, lm_id] > 40.0:
                        #     lm_color = (1,0,0)
                        cv2.circle(disp, tuple(lm[lm_id].astype(int).clip(0, disp.shape[0]-1)), radius=radius, color=lm_color, thickness=-1, lineType=cv2.LINE_AA)
                        if lm_confs is not None:
                            max_radius = img_size * 0.05
                            try:
                                conf_radius = max(2, int((1-lm_confs[img_id, lm_id]) * max_radius))
                            except ValueError:
                                conf_radius = 2
                            # if lm_confs[img_id, lm_id] > 0.4:
                            cirle_color = (0,0,255)
                            # if lm_confs[img_id, lm_id] < is_good_landmark(lm_confs, lm_rec_errs):
                            # if not is_good_landmark(lm_confs[img_id, lm_id], lm_rec_errs[img_id, lm_id]):
                            if lm_errs[img_id, lm_id] > 10.0:
                                cirle_color = (255,0,0)
                            cv2.circle(disp, tuple(lm[lm_id].astype(int)), conf_radius, cirle_color, 1, lineType=cv2.LINE_AA)

            # Draw outline if we actually have 68 valid landmarks.
            # Landmarks can be zeros for UMD landmark format (21 points).
            if draw_wireframe:
                nlms = (np.count_nonzero(lm.sum(axis=1)))
                if nlms == 68:
                    draw_wireframe_lines(disp, lm)
                elif nlms == 98:
                    draw_wireframe_lines_98(disp, lm)
        else:
            # colors = ['tab:gray', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:red', 'tab:blue']
            # colors_rgb = list(map(plt_colors.to_rgb, colors))

            colors = sns.color_palette("Set1", n_colors=14)
            for i in range(0,len(lm)):
                cv2.circle(disp, tuple(lm[i].astype(int)), radius=radius, color=colors[i], thickness=2, lineType=cv2.LINE_AA)
    return new_images


def draw_z(z_vecs):

    fy = 1
    width = 10
    z_zoomed = []
    for lvl, _ft in enumerate(to_numpy(z_vecs)):
        # _ft = (_ft-_ft.min())/(_ft.max()-_ft.min())
        vmin = 0 if lvl == 0 else -1

        canvas = np.zeros((int(fy*len(_ft)), width, 3))
        canvas[:int(fy*len(_ft)), :] = color_map(cv2.resize(_ft.reshape(-1,1), dsize=(width, int(fy*len(_ft))),
                                                            interpolation=cv2.INTER_NEAREST), vmin=-1.0, vmax=1.0)
        z_zoomed.append(canvas)
    return make_grid(z_zoomed, nCols=len(z_vecs), padsize=1, padval=0).transpose((1,0,2))


def overlay_heatmap(img, hm, heatmap_opacity=0.45):
    img_dtype = img.dtype
    img_new = img.copy()
    if img_new.dtype == np.uint8:
        img_new = img_new.astype(np.float32) / 255.0

    # hm = np.clip(hm, a_min=0, a_max=1.0)
    hm_colored = color_map(hm**3, vmin=0, vmax=1.0, cmap=plt.cm.inferno)
    if len(hm.shape) > 2:
        mask = cv2.blur(hm, ksize=(3, 3))
        print('mask', mask.dtype)
        mask = mask.mean(axis=2)
        mask = mask > 0.05
        for c in range(3):
            # img_new[...,c] = img[...,c] + hm[...,c]
            img_new[..., c][mask] = img[..., c][mask] * 0.7 + hm[..., c][mask] * 0.3
    else:
        # mask = mask > 0.05
        # img_new[mask] = img[mask] * 0.7 + hm_col[mask] * 0.3
        # heatmap_opacity = 0.7
        if hm_colored.shape != img.shape:
            hm_colored = cv2.resize(hm_colored, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        img_new = img_new + hm_colored * heatmap_opacity

    img_new = img_new.clip(0, 1)
    if img_dtype == np.uint8:
        img_new = cvt32FtoU8(img_new)
    assert img_new.dtype == img.dtype
    assert img_new.shape == img.shape
    return img_new


def draw_z_vecs(levels_z, ncols, size, vmin=-1, vmax=1, vertical=False):
    z_fy = 1.0
    width, height = [int(x) for x in size[:2]]
    def draw_z(z):
        z_zoomed = []
        for lvl, ft in enumerate(z):
            _ft = ft[:]
            # _ft = (_ft-_ft.min())/(_ft.max()-_ft.min())
            # canvas = np.zeros((height, width, 3))
            if vertical:
                _ft_reshaped = _ft.reshape(-1, 1)
            else:
                _ft_reshaped = _ft.reshape(1, -1)

            canvas = color_map(
                cv2.resize(_ft_reshaped, dsize=(width, height), interpolation=cv2.INTER_NEAREST),
                vmin=vmin,
                vmax=vmax
            )

            z_zoomed.append(canvas)
        return z_zoomed

    # pivots not used anymore FIXME: remove
    def draw_pivot(z_imgs, pivot):
        z_imgs_new = to_disp_images(z_imgs)
        for new_img in z_imgs_new:
            y = int(pivot*z_fy)
            cv2.line(new_img, (0, y), (new_img.shape[1], y), (1, 1, 1), thickness=1)
        return z_imgs_new

    # pivots = [z.shape[1] for z in levels_z if z is not None]
    # z_vis_list_per_level = [draw_pivot(draw_z(z), p) for z,p in zip(levels_z, pivots) if z is not None]
    z_vis_list_per_level = [draw_z(z) for z in levels_z if z is not None]
    z_grid_per_sample = [make_grid(all_vis_sample, nCols=len(levels_z)) for all_vis_sample in zip(*z_vis_list_per_level)]
    return make_grid(z_grid_per_sample, nCols=ncols, normalize=False)


def draw_status_bar(text, status_bar_width, status_bar_height, dtype=np.float32, text_size=-1, text_color=(255,255,255)):
    img_status_bar = np.zeros((status_bar_height, status_bar_width, 3), dtype=dtype)
    if text_size <= 0:
        text_size = status_bar_height * 0.025
    cv2.putText(img_status_bar, text, (4,img_status_bar.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1, cv2.LINE_AA)
    return img_status_bar



