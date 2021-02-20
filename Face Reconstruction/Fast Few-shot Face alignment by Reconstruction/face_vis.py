import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

from csl_common.utils.nn import to_numpy
from csl_common.vis import vis
from landmarks import lmconfig as lmcfg


def draw_results(X_resized, X_recon, levels_z=None, landmarks=None, landmarks_pred=None,
                 cs_errs=None, ncols=15, fx=0.5, fy=0.5, additional_status_text=''):

    clean_images = True
    if clean_images:
        landmarks=None

    nimgs = len(X_resized)
    ncols = min(ncols, nimgs)
    img_size = X_recon.shape[-1]

    disp_X = vis.to_disp_images(X_resized, denorm=True)
    disp_X_recon = vis.to_disp_images(X_recon, denorm=True)

    # reconstruction error in pixels
    l1_dists = 255.0 * to_numpy((X_resized - X_recon).abs().reshape(len(disp_X), -1).mean(dim=1))

    # SSIM errors
    ssim = np.zeros(nimgs)
    for i in range(nimgs):
        ssim[i] = compare_ssim(disp_X[i], disp_X_recon[i], data_range=1.0, multichannel=True)

    landmarks = to_numpy(landmarks)
    cs_errs = to_numpy(cs_errs)

    text_size = img_size/256
    text_thickness = 2

    #
    # Visualise resized input images and reconstructed images
    #
    if landmarks is not None:
        disp_X = vis.add_landmarks_to_images(disp_X, landmarks, draw_wireframe=False, landmarks_to_draw=lmcfg.LANDMARKS_19)
        disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, landmarks, draw_wireframe=False, landmarks_to_draw=lmcfg.LANDMARKS_19)

    if landmarks_pred is not None:
        disp_X = vis.add_landmarks_to_images(disp_X, landmarks_pred, color=(1, 0, 0))
        disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, landmarks_pred, color=(1, 0, 0))

    if not clean_images:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, l1_dists, format_string='{:.1f}',
                                           size=text_size, thickness=text_thickness)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', format_string='{:>4.2f}',
                                               vmax=0.8, vmin=0.2, size=text_size, thickness=text_thickness)
        if cs_errs is not None:
            disp_X_recon = vis.add_error_to_images(disp_X_recon, cs_errs, loc='bl-2', format_string='{:>4.2f}',
                                                   vmax=0.0, vmin=0.4, size=text_size, thickness=text_thickness)

    # landmark errors
    lm_errs = np.zeros(1)
    if landmarks is not None:
        try:
            from landmarks import lmutils
            lm_errs = lmutils.calc_landmark_nme_per_img(landmarks, landmarks_pred)
            disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs, loc='br', format_string='{:>5.2f}', vmax=15,
                                                   size=img_size/256, thickness=2)
        except:
            pass

    img_input = vis.make_grid(disp_X, nCols=ncols, normalize=False)
    img_recon = vis.make_grid(disp_X_recon, nCols=ncols, normalize=False)
    img_input = cv2.resize(img_input, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    img_recon = cv2.resize(img_recon, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    img_stack = [img_input, img_recon]

    #
    # Visualise hidden layers
    #
    VIS_HIDDEN = False
    if VIS_HIDDEN:
        img_z = vis.draw_z_vecs(levels_z, size=(img_size, 30), ncols=ncols)
        img_z = cv2.resize(img_z, dsize=(img_input.shape[1], img_z.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_stack.append(img_z)

    cs_errs_mean = np.mean(cs_errs) if cs_errs is not None else np.nan
    status_bar_text = ("l1 recon err: {:.2f}px  "
                       "ssim: {:.3f}({:.3f})  "
                       "lms err: {:2} {}").format(
        l1_dists.mean(),
        cs_errs_mean,
        1 - ssim.mean(),
        lm_errs.mean(),
        additional_status_text
    )

    img_status_bar = vis.draw_status_bar(status_bar_text,
                                         status_bar_width=img_input.shape[1],
                                         status_bar_height=30,
                                         dtype=img_input.dtype)
    img_stack.append(img_status_bar)

    return np.vstack(img_stack)