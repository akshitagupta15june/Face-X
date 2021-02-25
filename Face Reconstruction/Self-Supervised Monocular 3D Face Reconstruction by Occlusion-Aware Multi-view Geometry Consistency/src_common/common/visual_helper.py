
# system
from __future__ import print_function

# python lib
import numpy as np

# tf_render
import tensorflow as tf

# visual lm2d
def draw_landmark_image(list_img, list_lm, img_height, img_width, color):
    """
    :param list_img:
    :param list_lm:
    :param img_height:
    :param img_width:
    :param color: 1:r 2:g 3:b
    :return:
    """
    list_img_lm = []
    for i in range(len(list_img)):
        img = list_img[i]

        if len(list_img) == len(list_lm):
            lm = list_lm[i]
        else:
            lm = list_lm[0]
        img_draw_lm = img

        img_draw_lm = tf.image.convert_image_dtype(img_draw_lm, dtype=tf.float32)
        img_draw_lm = render_lm2d_circle_image(img_draw_lm, lm, img_height, img_width, color=color)
        img_draw_lm = tf.image.convert_image_dtype(img_draw_lm, dtype=tf.uint8)
        list_img_lm.append(img_draw_lm)
    return list_img_lm

def render_lm2d(lm2d_batch_xy, h, w):
    """
    :param lm2d_batch:
    :param h:
    :param w:
    :return:
    """
    # preprocess
    """
    row correspond to y
    column correspond to x
    (row, column) = (y, x)
    """
    x = lm2d_batch_xy[:, :, 0]
    y = lm2d_batch_xy[:, :, 1]
    x = tf.clip_by_value(x, 0, w-1)
    y = tf.clip_by_value(y, 0, h-1)
    lm2d_batch = tf.stack([y, x], axis=-1)

    #
    visual_lm2d = []
    for b_it in range(lm2d_batch.shape[0]):
        lm2d = lm2d_batch[b_it]
        lm2d = tf.cast(lm2d, dtype=tf.int64)
        r = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.linspace(0.0, 1.0, lm2d.shape[0]),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        g = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.zeros(shape=[lm2d.shape[0]], dtype=tf.float32),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        b = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.zeros(shape=[lm2d.shape[0]], dtype=tf.float32),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        rgb = tf.stack([r, g, b], axis=-1)
        visual_lm2d.append(rgb)

    visual_lm2d = tf.stack(visual_lm2d)
    return visual_lm2d

def render_lm2d_image(image, lm2d_batch_xy, h, w, color=2, radius=1, light=1.0):
    """
    :param image: (0, 1)
    :param lm2d_batch_xy:
    :param h:
    :param w:
    :param color:
    :param radius:
    :param light:
    :return:
    """
    # preprocess
    """
    row correspond to y
    column correspond to x
    (row, column) = (y, x)
    """
    x = lm2d_batch_xy[:, :, 0]
    y = lm2d_batch_xy[:, :, 1]
    x = tf.clip_by_value(x, 0, w-1)
    y = tf.clip_by_value(y, 0, h-1)
    lm2d_batch = tf.stack([y, x], axis=-1)

    """
    circle lm    
    """
    #
    visual_lm2d = []
    for b_it in range(lm2d_batch.shape[0]):
        lm2d = lm2d_batch[b_it]
        lm2d = tf.cast(lm2d, dtype=tf.int64)
        if color == 1:
            r = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.linspace(light, 1.0, tf.shape(lm2d)[0]),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        else:
            r = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.zeros(shape=[tf.shape(lm2d)[0]], dtype=tf.float32),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        if color == 2:
            g = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.linspace(light, 1.0, tf.shape(lm2d)[0]),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        else:
            g = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.zeros(shape=[tf.shape(lm2d)[0]], dtype=tf.float32),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        if color == 3:
            b = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.linspace(light, 1.0, tf.shape(lm2d)[0]),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        else:
            b = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.zeros(shape=[tf.shape(lm2d)[0]], dtype=tf.float32),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        rgb = tf.stack([r, g, b], axis=-1)
        visual_lm2d.append(rgb)
    visual_lm2d = tf.stack(visual_lm2d)

    """
    assign image
    """
    # Mask
    mask_lm2d = []
    for b_it in range(lm2d_batch.shape[0]):
        lm2d = lm2d_batch[b_it]
        lm2d = tf.cast(lm2d, dtype=tf.int64)
        r = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.ones(shape=[tf.shape(lm2d)[0]], dtype=tf.float32),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        g = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.ones(shape=[tf.shape(lm2d)[0]], dtype=tf.float32),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        b = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.ones(shape=[tf.shape(lm2d)[0]], dtype=tf.float32),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        rgb = tf.stack([r, g, b], axis=-1)
        mask_lm2d.append(rgb)
    mask_lm2d = tf.stack(mask_lm2d)
    mask_lm2d = 1.0-mask_lm2d

    visual_image = image
    visual_image = visual_image * mask_lm2d

    visual_image = visual_image + visual_lm2d
    return visual_image

def lm_expand_circle(lm2d_batch, h, w):
        batch_size = lm2d_batch.shape[0]
        num_lm = tf.shape(lm2d_batch)[1]

        neighboor = tf.constant(
            [[-1., -1.], [-1., 0.], [-1., 1.],
             [0., -1.], [0., 0.], [0., 1.],
             [1., -1.], [1., 0.], [1., 1.],
             ]
        )
        # neighboor = tf.expand_dims(neighboor, 0)
        # neighboor = tf.tile(neighboor, [batch_size, num_lm, 1])
        #
        # lm2d_batch = tf.tile(lm2d_batch, [1, 9, 1])
        # lm_neightboor = tf.add(neighboor, lm2d_batch)
        # y = lm_neightboor[:, :, 0]
        # y = tf.clip_by_value(y, 0, h-1)
        # x = lm_neightboor[:, :, 1]
        # x = tf.clip_by_value(x, 0, w-1)
        # lm2d_point_batch = tf.stack([y, x], axis=-1)

        neighboor = tf.expand_dims(neighboor, 0)
        neighboor = tf.tile(neighboor, [batch_size, 1, 1])
        neighboor = tf.transpose(neighboor, perm=[1, 0, 2])

        lm2d_circle_batch = []
        for i in range(lm2d_batch.shape[1]):
            lm_neightboor = tf.add(neighboor, lm2d_batch[:, i, :])
            lm_neightboor = tf.transpose(lm_neightboor, perm=[1, 0, 2])
            y = lm_neightboor[:, :, 0]
            y = tf.clip_by_value(y, 0, h-1)
            x = lm_neightboor[:, :, 1]
            x = tf.clip_by_value(x, 0, w-1)
            lm2d_point_batch = tf.stack([y, x], axis=-1)
            if i == 0:
                lm2d_circle_batch = lm2d_point_batch
            else:
                lm2d_circle_batch = tf.concat([lm2d_circle_batch, lm2d_point_batch], axis=1)
        return lm2d_circle_batch

def render_lm2d_circle_image(image, lm2d_batch_xy, h, w, color=2, radius=1, light=1.0):
    """
    :param image: (0, 1)
    :param lm2d_batch_xy:
    :param h:
    :param w:
    :param color:
    :param radius:
    :param light:
    :return:
    """
    # preprocess
    """
    row correspond to y
    column correspond to x
    (row, column) = (y, x)
    """
    x = lm2d_batch_xy[:, :, 0]
    y = lm2d_batch_xy[:, :, 1]
    x = tf.clip_by_value(x, 0, w-1)
    y = tf.clip_by_value(y, 0, h-1)
    lm2d_batch = tf.stack([y, x], axis=-1)

    """
    circle lm    
    """
    lm2d_batch = lm_expand_circle(lm2d_batch, h, w)

    #
    visual_lm2d = []
    for b_it in range(lm2d_batch.shape[0]):
        lm2d = lm2d_batch[b_it]
        lm2d = tf.cast(lm2d, dtype=tf.int64)
        if color == 1:
            r = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.linspace(light, 1.0, lm2d.shape[0]),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        else:
            r = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.zeros(shape=[lm2d.shape[0]], dtype=tf.float32),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        if color == 2:
            g = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.linspace(light, 1.0, lm2d.shape[0]),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        else:
            g = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.zeros(shape=[lm2d.shape[0]], dtype=tf.float32),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        if color == 3:
            b = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.linspace(light, 1.0, lm2d.shape[0]),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        else:
            b = tf.sparse_tensor_to_dense(
                tf.SparseTensor(indices=lm2d, values=tf.zeros(shape=[lm2d.shape[0]], dtype=tf.float32),
                                dense_shape=[h, w]),
                validate_indices=False
            )
        rgb = tf.stack([r, g, b], axis=-1)
        visual_lm2d.append(rgb)
    visual_lm2d = tf.stack(visual_lm2d)

    """
    assign image
    """
    # Mask
    mask_lm2d = []
    for b_it in range(lm2d_batch.shape[0]):
        lm2d = lm2d_batch[b_it]
        lm2d = tf.cast(lm2d, dtype=tf.int64)
        r = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.ones(shape=[lm2d.shape[0]], dtype=tf.float32),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        g = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.ones(shape=[lm2d.shape[0]], dtype=tf.float32),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        b = tf.sparse_tensor_to_dense(
            tf.SparseTensor(indices=lm2d, values=tf.ones(shape=[lm2d.shape[0]], dtype=tf.float32),
                            dense_shape=[h, w]),
            validate_indices=False
        )
        rgb = tf.stack([r, g, b], axis=-1)
        mask_lm2d.append(rgb)
    mask_lm2d = tf.stack(mask_lm2d)
    mask_lm2d = 1.0-mask_lm2d

    visual_image = image
    visual_image = visual_image * mask_lm2d

    visual_image = visual_image + visual_lm2d
    return visual_image

# visual heatmap
def gauss(x, a, b, c, d=0):
    return a * tf.exp(-(x - b)**2 / (2 * c**2)) + d

def pixel_error_heatmap(image_error):
    """
    :param image_error: shape=[bs, h, w, 1], [0, 1]
    :return:
    """
    x = image_error
    # x = tf.reduce_max(tf.reshape(x, [x.shape[0], -1]), axis=1)
    # x = tf.divide(x, tf.reshape(v_error_max, [x.shape[0], 1, 1, 1]) + 1e-6)

    if len(image_error.shape) == 3:
        x = tf.expand_dims(image_error, -1)


    color_0 = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color_1 = gauss(x, 1, .5, .3)
    color_2 = gauss(x, 1, .2, .3)
    color = tf.concat([color_0, color_1, color_2], axis=3)

    color = tf.clip_by_value(color, 0.0, 1.0)

    return color

# net image / visual image
def preprocess_image(image):
    # Assuming input image is uint8
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image #* 2. - 1.

def deprocess_image_series(list_image):
    list_image_depro = []
    for i in range(len(list_image)):
        image = list_image[i]
        image_depro = deprocess_image(image)
        list_image_depro.append(image_depro)
    return list_image_depro

def deprocess_normal_series(list_image):
    list_image_depro = []
    for i in range(len(list_image)):
        image = list_image[i]
        image = image / 2.0 + 0.5
        image_depro = deprocess_image(image)
        list_image_depro.append(image_depro)
    return list_image_depro

def deprocess_image(image):
    # Assuming input image is float32
    batch_size = image.shape[0]
    # norm

    # image_max = tf.reduce_max(
    #     tf.reshape(image, [batch_size, -1]), axis=1)
    # image_norm = tf.divide(image,
    #                          tf.reshape(image_max, [batch_size, 1, 1, 1]) + 1e-6)
    image = tf.clip_by_value(image, 0.0, 1.0)
    #image = (image + 1.) / 2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

def deprocess_gary_image_series(list_image, convert=True):
    if isinstance(list_image, list) == False:
        list_image = [list_image]

    list_image_depro = []
    for i in range(len(list_image)):
        image = list_image[i]
        image_depro = deprocess_gary_image(image, convert)
        list_image_depro.append(image_depro)
    return list_image_depro

def deprocess_gary_image(image, convert=True):
    # Assuming input image is float32
    image = tf.image.grayscale_to_rgb(image)
    if convert:
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

    return image

# multi-view image concat/insert
def concate_image_series(list_image_l, list_image_r, axis):
    list_cat = []
    for i in range(len(list_image_l)):
        image_l = list_image_l[i]
        image_r = list_image_r[i]
        image_cat = tf.concat([image_l, image_r], axis=axis)
        list_cat.append(image_cat)
    return list_cat

def insert_semi_image_series(list_tar, list_src):
    list_cat = []
    for i in range(len(list_src)):
        if i == len(list_src)/2:
            list_cat.append(list_tar[0])
        list_cat.append(list_src[i])
    list_cat = tf.concat(list_cat, axis=2) # bs, h, w
    return list_cat

def concate_semi_image_series(list, list_src=None):
    if list_src is None:
        list_tar = [list[0]]
        list_src = list[1:]
    else:
        list_tar = list
        list_src = list_src
    list_cat = []
    for i in range(len(list_src)):
        if i == len(list_src)/2:
            list_cat.append(list_tar[0])
        list_cat.append(list_src[i])
    if isinstance(list_cat[0], np.ndarray):
        list_cat = np.concatenate(list_cat, axis=2)  # bs, h, w
    else:
        list_cat = tf.concat(list_cat, axis=2) # bs, h, w
    return list_cat

# visual depthmap
def normal_max_for_show(disp):
    disp_max = tf.reduce_max(disp)
    disp_new = disp/disp_max
    disp_new = disp_new*255
    disp_new_uint8 = tf.cast(disp_new, dtype=tf.uint8)
    return disp_new_uint8

def normal_depthmap_for_show_bgMax(disp):
    #disp_min = tf.contrib.distributions.percentile(disp, q=0, axis=[1, 2], interpolation='lower')
    disp_min = tf.reduce_min(disp)
    #disp_max = disp_min+255
    #disp = tf.clip_by_value(disp, disp_min, disp_max)
    #disp_max = tf.expand_dims(tf.expand_dims(disp_max, 1), 1)
    disp_min = tf.reshape(disp_min, [1, 1, 1, 1])
    disp_new = disp-disp_min
    disp_new_uint8 = tf.cast(disp_new, dtype=tf.uint8)
    return disp_new_uint8

def normal_depthmap_for_show(disp):
    disp_max = tf.contrib.distributions.percentile(disp, q=100, axis=[1, 2], interpolation='lower')
    disp_max = tf.reduce_max(disp_max)
    disp_min = disp_max-255*2
    disp = tf.clip_by_value(disp, disp_min, disp_max)
    #disp_max = tf.expand_dims(tf.expand_dims(disp_max, 1), 1)
    disp_min = tf.reshape(disp_min, [1, 1, 1, 1])

    disp_new = disp-disp_min
    #disp_new = (disp_new - disp_min) / (disp_max - disp_min)

    # disp_new = []
    # for i in range(disp.shape[0]):
    #     #disp_i = tf_render.clip_by_value(disp[i], disp_min[i], disp_max[i])
    #     dn = (disp[i] - disp_min[i]) / (disp_max[i] - disp_min[i])
    #     disp_new.append(dn)
    # disp_new = tf_render.stack(disp_new)
    disp_new = tf.cast(disp_new, dtype=tf.uint8)
    return disp_new
