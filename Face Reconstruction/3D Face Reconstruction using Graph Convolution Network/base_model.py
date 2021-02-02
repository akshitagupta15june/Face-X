import logging
import math
import os
import shutil
import time

import cv2
import imageio
import numpy as np
import scipy.sparse
import tensorflow as tf

import utils
from lib import graph, mesh_renderer
from lib.mesh_io import write_obj

logger = logging.getLogger('x')


class BaseModel():
  """
    Mesh Convolutional Autoencoder which uses the Chebyshev approximation.
  """

  def __init__(self, args, sess, graphs, refer_mesh, image_paths, img_file):
    self.sess = sess
    self.graph = graphs
    mesh_shape = list(refer_mesh['vertices'].shape)
    self.gan = args.gan
    self.wide = args.wide
    self.root_dir = args.root_dir
    self.img_file = img_file
    self.stage = args.stage
    if args.mode == 'test':
      self.restore = True
    else:
      self.restore = args.restore

    self.laplacians, self.downsamp_trans, self.upsamp_trans, self.pool_size = utils.init_sampling(
        refer_mesh, os.path.join(args.root_dir, 'data', 'params', args.name), args.name)
    logger.info("Transform Matrices and Graph Laplacians Generated.")
    self.refer_meshes = utils.get_mesh_list(args.name)

    self.bfm = utils.BFM_model(self.root_dir, 'data/bfm2009_face.mat')
    # color = np.ones_like(refer_mesh['vertices'], dtype=np.uint8)
    # color[self.bfm.skin_index] = 0
    # write_obj('tests.obj', refer_mesh['vertices'], refer_mesh['faces'], color)
    # write_obj('test.obj', refer_mesh['vertices'], refer_mesh['faces'], color)

    self.buffer_size = args.buffer_size
    self.workers = args.workers
    self.num_filter = [16, 16, 16, 32]
    self.c_k = 6
    self.cam_z = 34
    self.z_dim = args.nz
    self.num_vert = mesh_shape[0]
    self.vert_dim = 6
    self.drop_rate = args.drop_rate
    self.batch_size = args.batch_size
    self.num_epochs = args.epoch
    self.img_size = args.img_size
    self.learning_rate = args.lr
    self.adv_lambda = args.adv_lambda
    if args.suffix is None:
      self.dir_name = args.name
    else:
      self.dir_name = args.name + '_' + args.suffix
    self.brelu = self.b1relu
    self.pool = self.poolwT
    self.unpool = self.poolwT

    self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (5, 5)).astype(np.float32)[..., np.newaxis]
    self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (9, 9)).astype(np.float32)[..., np.newaxis]
    # lm_3d_idx = [
    #     int(x)
    #     for x in open('data/face_landmarks.txt', 'r').readlines()
    #     if len(x.strip()) > 1
    # ]
    # # self.lm_3d_idx = lm_3d_idx[8:9] + lm_3d_idx[17:]
    # self.lm_3d_idx = lm_3d_idx[17:]
    self.lm_3d_idx = self.bfm.landmark[17:]

    train_image_paths, self.val_image_paths, self.test_image_paths = utils.make_paths(
        image_paths, os.path.join(self.root_dir, 'data', 'params', args.name, 'image'),
        self.root_dir)
    self.train_image_paths = np.array(train_image_paths, dtype='object')

    num_train = len(self.train_image_paths)
    logger.info('Number of train data: %d', num_train)
    self.num_batches = num_train // self.batch_size
    if args.eval == 0:
      self.eval_frequency = self.num_batches
    elif args.eval < 1:
      self.eval_frequency = int(self.num_batches * args.eval)
    else:
      self.eval_frequency = int(args.eval)
    logger.info('Evaluation frequency: %d', self.eval_frequency)

    self.vert_mean = np.reshape(self.bfm.shapeMU, [-1, 3])

    self.decay_steps = num_train // args.batch_size

    self.regularizers = []
    self.regularization = 5e-4

    self.ckpt_dir = os.path.join('checkpoints', self.dir_name)
    self.summ_dir = os.path.join('summaries', self.dir_name)
    self.samp_dir = os.path.join('samples', self.dir_name)

    self.build_graph()

  def build_graph(self):
    """Build the computational graph of the model."""
    # self.graph = tf.Graph()

    # with self.graph.as_default():
    # Inputs.
    with tf.name_scope('inputs'):

      data_idxs = [x for x in range(len(self.train_image_paths))]
      image_dataset = tf.data.Dataset.from_tensor_slices(data_idxs)
      # image_dataset = image_dataset.map(
      #     lambda start_idx: tf.py_func(self.load_image_bin, [start_idx], [tf.float32, tf.float32]))
      image_dataset = image_dataset.map(
          lambda start_idx: tf.py_func(self.load_image_bin, [start_idx], tf.float32))

      image_dataset = image_dataset.shuffle(buffer_size=self.buffer_size)
      image_dataset = image_dataset.batch(self.batch_size)
      image_dataset = image_dataset.repeat()
      image_iterator = image_dataset.make_one_shot_iterator()
      # self.train_rgbas, self.train_2dlms = image_iterator.get_next()
      self.train_rgbas = image_iterator.get_next()
      self.train_rgbas.set_shape([self.batch_size, self.img_size, self.img_size, 4])
      self.train_images = (self.train_rgbas[..., :3] + 1) * 127.5
      # self.train_2dlms.set_shape([self.batch_size, len(self.lm_3d_idx), 2])

      self.refer_faces = [
          tf.convert_to_tensor(x['faces'], dtype=tf.int32, name='refer_faces_{}'.format(i))
          for i, x in enumerate(self.refer_meshes)
      ]

      self.ph_rgbas = tf.placeholder(tf.float32, (self.batch_size, self.img_size, self.img_size, 4),
                                     'input_rgbas')
      self.input_images = (self.ph_rgbas[..., :3] + 1) * 127.5
      # self.input_images = tf.floor((self.ph_rgbas[..., 2::-1] + 1) * 127.5)

      self.ph_2dlms = tf.placeholder(tf.float32, (self.batch_size, len(self.lm_3d_idx), 2),
                                     'input_2dlm')
      self.ph_ren_lambda = tf.placeholder(tf.float32, (), 'render_lambda')
      self.ph_ref_lambda = tf.placeholder(tf.float32, (), 'refine_lambda')
      # self.ph_adv_lambda = tf.placeholder(tf.float32, (), 'adv_lambda')

    with tf.gfile.GFile(os.path.join(self.root_dir, 'data/FaceReconModel.pb'), 'rb') as f:
      face_rec_graph_def = tf.GraphDef()
      face_rec_graph_def.ParseFromString(f.read())

    def get_emb_coeff(net_name, inputs):
      resized = tf.image.resize_images(inputs, [224, 224])
      bgr_inputs = resized[..., ::-1]
      tf.import_graph_def(face_rec_graph_def, name=net_name, input_map={'input_imgs:0': bgr_inputs})
      image_emb = self.graph.get_tensor_by_name(net_name + '/resnet_v1_50/pool5:0')
      image_emb = tf.squeeze(image_emb, axis=[1, 2])
      coeff = self.graph.get_tensor_by_name(net_name + '/coeff:0')
      return image_emb, coeff

    image_emb, self.coeff = get_emb_coeff('facerec', self.train_images)
    image_emb_test, self.coeff_test = get_emb_coeff('facerec_test', self.input_images)

    with tf.gfile.GFile(os.path.join(self.root_dir, 'data/FaceNetModel.pb'), 'rb') as f:
      face_net_graph_def = tf.GraphDef()
      face_net_graph_def.ParseFromString(f.read())

    def get_img_feat(net_name, inputs):
      # inputs should be in [0, 255]
      # facenet_input = tf.image.resize_image_with_crop_or_pad(inputs, 160, 160)
      # TODO: fix resize issue!!!
      facenet_input = tf.image.resize_images(inputs, [160, 160])

      facenet_input = (facenet_input - 127.5) / 128.0
      tf.import_graph_def(face_net_graph_def, name=net_name, input_map={
          'input:0': facenet_input,
          'phase_train:0': False
      })
      image_feat = self.graph.get_tensor_by_name(
          net_name + '/InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0')
      image_feat = tf.squeeze(image_feat, axis=[1, 2])
      return image_feat

    image_feat = get_img_feat('facenet', self.train_images)
    image_feat_test = get_img_feat('facenet_test', self.input_images)

    self.image_emb = tf.concat([image_emb, image_feat], axis=-1)
    self.image_emb_test = tf.concat([image_emb_test, image_feat_test], axis=-1)

    pred_results = self.inference(self.train_rgbas, self.coeff, self.image_emb)
    self.vert_pred = pred_results['vertice']
    self.pca_text_pred = pred_results['pca_texture']
    self.gcn_text_pred = pred_results['gcn_texture']
    self.pca_color_pred = pred_results['pca_color']
    self.gcn_color_pred = pred_results['gcn_color']
    self.proj_color_pred = pred_results['proj_color']
    self.pca_render_pred = pred_results['pca_render_color']
    self.gcn_render_pred = pred_results['gcn_render_color']
    self.lm_proj_pred = pred_results['lm_project']
    # render_mask = self._erosion2d(self.train_rgbas[..., 3:])
    render_mask = self.pca_render_pred[..., 3:] * self.train_rgbas[..., 3:]
    gcn_render_image = (self.gcn_render_pred[..., :3] + 1) * 127.5
    self.gcn_overlay = gcn_render_image[..., :3] * render_mask +\
        self.train_images[..., :3] * (1 - render_mask)
    gcn_image_feat = get_img_feat('facenet_gcn', self.gcn_overlay)
    self.all_loss, self.pca_loss, self.gcn_loss, self.proj_loss, self.refine_loss, self.perc_loss, self.var_loss, self.sym_loss = self.compute_loss(
        self.train_rgbas, self.pca_render_pred, self.gcn_render_pred, self.pca_text_pred,
        self.gcn_text_pred, self.proj_color_pred, self.pca_color_pred, self.gcn_color_pred,
        image_feat, gcn_image_feat, self.regularization)

    test_results = self.inference(self.ph_rgbas, self.coeff_test, self.image_emb_test,
                                  is_training=False, reuse=True, get_inter=True)
    self.vert_test = test_results['vertice']
    self.norm_test = test_results['normal']
    self.pca_text_test = test_results['pca_texture']
    self.gcn_text_test = test_results['gcn_texture']
    self.pca_color_test = test_results['pca_color']
    self.gcn_color_test = test_results['gcn_color']
    self.proj_color_test = test_results['proj_color']
    self.pca_ren_tex_test = test_results['pca_render_text']
    self.gcn_ren_tex_test = test_results['gcn_render_text']
    self.pca_ren_clr_test = test_results['pca_render_color']
    self.gcn_ren_clr_test = test_results['gcn_render_color']
    self.lm_proj_test = test_results['lm_project']
    # render_mask_test = self._erosion2d(self.ph_rgbas[..., 3:])
    render_mask_test = self.pca_ren_clr_test[..., 3:] * self.ph_rgbas[..., 3:]
    gcn_ren_image_test = (self.gcn_ren_clr_test[..., :3] + 1) * 127.5
    self.gcn_over_test = gcn_ren_image_test[..., :3] * render_mask_test +\
        self.input_images[..., :3] * (1 - render_mask_test)
    gcn_image_feat_test = get_img_feat('facenet_gcn_test', self.gcn_over_test)
    self.test_all_loss, self.test_pca_loss, self.test_gcn_loss, self.test_proj_loss, self.test_refine_loss, self.test_perc_loss, _, _ = self.compute_loss(
        self.ph_rgbas, self.pca_ren_clr_test, self.gcn_ren_clr_test, self.pca_text_test,
        self.gcn_text_test, self.proj_color_test, self.pca_color_test, self.gcn_color_test,
        image_feat_test, gcn_image_feat_test, self.regularization, True)

    self.d_loss = None
    if self.gan:
      real_image = self.train_rgbas[..., :3]
      fake_image = self.gcn_overlay / 127.5 - 1.0
      self.g_loss, self.d_loss = self.compute_gan_loss(real_image, fake_image)
      self.all_loss = self.all_loss + self.g_loss

      real_img_test = self.ph_rgbas[..., :3]
      fake_img_test = self.gcn_over_test / 127.5 - 1.0
      self.test_g_loss, self.test_d_loss = self.compute_gan_loss(real_img_test, fake_img_test,
                                                                 reuse=True)
      self.test_all_loss = self.test_all_loss + self.test_g_loss

    self.gen_train, self.dis_train = self.training(self.all_loss, self.d_loss)
    # self.op_encoder = self.encoder(self.ph_data, reuse=True)
    # self.op_decoder = self.decoder(self.ph_z, reuse=True)

    # Initialize variables, i.e. weights and biases.
    self.op_init = tf.global_variables_initializer()

    # Summaries for TensorBoard and Save for model parameters.
    self.op_summary = tf.summary.merge_all()

    var_all = tf.global_variables()
    trainable_vars = tf.trainable_variables()
    bn_vars = [x for x in var_all if 'BatchNorm/moving' in x.name]
    global_vars = [x for x in var_all if 'training' in x.name]
    vars_to_save = trainable_vars + bn_vars + global_vars
    self.op_saver = tf.train.Saver(var_list=vars_to_save, max_to_keep=3)

    logger.info('Successfully Build Graph')

  def inference(self, images, coeff, image_emb, is_training=True, reuse=False, get_inter=False):

    shape_coef, exp_coef, color_coef, angles, gamma, translation = utils.split_bfm09_coeff(coeff)

    # shapeMU = tf.constant(self.bfm.shapeMU, dtype=tf.float32)
    shapePC = tf.constant(self.bfm.shapePC, dtype=tf.float32)
    # expMU = tf.constant(self.bfm.expressionMU, dtype=tf.float32)
    expPC = tf.constant(self.bfm.expressionPC, dtype=tf.float32)
    colorMU = tf.constant(self.bfm.colorMU, dtype=tf.float32)
    colorPC = tf.constant(self.bfm.colorPC, dtype=tf.float32)

    vert_offset = tf.einsum('ij,aj->ai', shapePC, shape_coef) + tf.einsum(
        'ij,aj->ai', expPC, exp_coef)
    vertice = tf.reshape(vert_offset, [self.batch_size, self.num_vert, 3]) + self.vert_mean
    vertice = vertice - tf.reduce_mean(self.vert_mean, axis=0, keepdims=True)
    # normal = tf.nn.l2_normalize(vertice)
    normal = self.compute_norm(vertice)

    rotation = utils.rotation_matrix_tf(angles)
    vert_trans = tf.matmul(vertice, rotation) + tf.reshape(translation, [self.batch_size, 1, 3])
    normal_rot = tf.matmul(normal, rotation)

    pca_texture = tf.einsum('ij,aj->ai', colorPC, color_coef) + colorMU
    # outputs of pca is [0, 255]
    pca_texture = tf.clip_by_value(pca_texture, 0.0, 255.0)
    pca_texture = pca_texture / 127.5 - 1
    pca_texture = tf.reshape(pca_texture, [self.batch_size, self.num_vert, 3])

    # outputs of mesh_decoder using tanh for activation

    with tf.variable_scope('render', reuse=reuse):
      camera_position = tf.constant([0, 0, 10], dtype=tf.float32)
      camera_lookat = tf.constant([0, 0, 0], dtype=tf.float32)
      camera_up = tf.constant([0, 1, 0], dtype=tf.float32)
      light_positions = tf.tile(tf.reshape(tf.constant([0, 0, 0], dtype=tf.float32), [1, 1, 3]),
                                [self.batch_size, 1, 1])
      light_intensities = tf.tile(tf.reshape(tf.constant([0, 0, 0], dtype=tf.float32), [1, 1, 3]),
                                  [self.batch_size, 1, 1])
      fov_y = 12.5936
      ambient_color = tf.tile(tf.reshape(tf.constant([1, 1, 1], dtype=tf.float32), [1, 3]),
                              [self.batch_size, 1])

      def postprocess(inputs):
        outputs = tf.clip_by_value(inputs, 0.0, 1.0)
        outputs = outputs * [[[[2.0, 2.0, 2.0, 1.0]]]] - [[[[1.0, 1.0, 1.0, 0.0]]]]
        return outputs

      # make color between 0 and 1 before rendering
      # outputs will be post processed, [-1, 1] for rgb value
      def neural_renderer(vertices, triangles, normals, diffuse_colors):
        renders, shift_vert = mesh_renderer.mesh_renderer(
            vertices=vertices, triangles=triangles, normals=normals, diffuse_colors=diffuse_colors,
            camera_position=camera_position, camera_lookat=camera_lookat, camera_up=camera_up,
            light_positions=light_positions, light_intensities=light_intensities,
            image_width=self.img_size, image_height=self.img_size, fov_y=fov_y,
            ambient_color=ambient_color)
        return postprocess(renders), shift_vert

      pca_render_text, shift_vert = neural_renderer(vertices=vert_trans,
                                                    triangles=self.refer_faces[0],
                                                    normals=normal_rot,
                                                    diffuse_colors=(pca_texture + 1) / 2)
      pca_color = self.illumination((pca_texture + 1) / 2, normal_rot, gamma)

      pca_render_color, _ = neural_renderer(vertices=vert_trans, triangles=self.refer_faces[0],
                                            normals=normal_rot, diffuse_colors=pca_color)
      pca_color = pca_color * 2 - 1

    facial = tf.tan(fov_y / 360.0 * math.pi)
    facial = tf.reshape(facial, [-1, 1, 1])
    proj_vert = shift_vert[..., :3] * [[[1, -1, -1]]]
    proj_vert = proj_vert[..., :2] / facial / proj_vert[..., 2:3]

    eros_mask = self._erosion2d(images[..., 3:])
    eros_image = tf.concat([images[..., :3], eros_mask], axis=-1)
    lm_project = tf.gather(proj_vert, self.lm_3d_idx, axis=1)
    proj_color = self.project_color(proj_vert, eros_image)
    visiable = tf.cast(normal_rot[..., 2:3] > 0, tf.float32) * proj_color[..., 3:4]
    proj_color = tf.concat([proj_color[..., :3] * visiable, visiable], axis=-1)
    # TODO: 
    # refine_input = pca_texture
    # refine_input = tf.concat([pca_texture, proj_color[..., :3]], axis=-1)
    refine_input = tf.concat([pca_texture, proj_color], axis=-1)
    gcn_texture = self.mesh_generator(image_emb, refine_input, reuse=reuse)

    with tf.variable_scope('render', reuse=reuse):
      gcn_render_text, _ = neural_renderer(vertices=vert_trans, triangles=self.refer_faces[0],
                                           normals=normal_rot, diffuse_colors=(gcn_texture + 1) / 2)
      gcn_color = self.illumination((gcn_texture + 1) / 2, normal_rot, gamma)
      gcn_render_color, _ = neural_renderer(vertices=vert_trans, triangles=self.refer_faces[0],
                                            normals=normal_rot, diffuse_colors=gcn_color)
      gcn_color = gcn_color * 2 - 1

    tf.summary.image('pca_render_text', pca_render_text, max_outputs=4)
    tf.summary.image('gcn_render_text', gcn_render_text, max_outputs=4)
    tf.summary.image('pca_render_color', pca_render_color, max_outputs=4)
    tf.summary.image('gcn_render_color', gcn_render_color, max_outputs=4)

    logger.info('Successfully Inferenced')

    return {
        # 'vertice': vert_trans,
        'vertice': vertice,
        'normal': normal,
        'pca_texture': pca_texture,  # [-1, 1]
        'gcn_texture': gcn_texture,  # [-1, 1]
        'pca_color': pca_color,  # [-1, 1]
        'gcn_color': gcn_color,  # [-1, 1]
        'proj_color': proj_color,  # [-1, 1]
        'pca_render_text': pca_render_text,  # [-1, 1]
        'gcn_render_text': gcn_render_text,  # [-1, 1]
        'pca_render_color': pca_render_color,  # [-1, 1]
        'gcn_render_color': gcn_render_color,  # [-1, 1]
        'lm_project': lm_project
    }

  def compute_loss(self, input_image, pca_render, gcn_render, pca_texture, gcn_texture, proj_color,
                   pca_color, gcn_color, input_feat, gcn_feat, regularization, get_inter=False):
    """Adds to the inference model the layers required to generate loss."""
    with tf.name_scope('loss'):
      with tf.name_scope('data_loss'):
        skin_mask = self._erosion2d(input_image[..., 3:])
        gcn_render_mask = tf.round(gcn_render[..., 3:]) * skin_mask

        # pca_render_loss = tf.losses.mean_squared_error(
        pca_render_loss = tf.losses.absolute_difference(
            predictions=pca_render[..., :3] * gcn_render_mask, labels=input_image[..., :3] *
            gcn_render_mask, reduction=tf.losses.Reduction.SUM) / tf.reduce_sum(gcn_render_mask)

        # gcn_render_loss = tf.losses.mean_squared_error(
        gcn_render_loss = tf.losses.absolute_difference(
            predictions=gcn_render[..., :3] * gcn_render_mask, labels=input_image[..., :3] *
            gcn_render_mask, reduction=tf.losses.Reduction.SUM) / tf.reduce_sum(gcn_render_mask)

        # project_loss_image = tf.losses.mean_squared_error(
        project_loss_image = tf.losses.absolute_difference(
            predictions=gcn_color * proj_color[..., 3:],
            labels=proj_color[..., :3] * proj_color[..., 3:], reduction=tf.losses.Reduction.MEAN)

        # project_loss_pca = tf.losses.mean_squared_error(
        project_loss_pca = tf.losses.absolute_difference(
            predictions=gcn_color * (1 - proj_color[..., 3:]),
            labels=pca_color * (1 - proj_color[..., 3:]), reduction=tf.losses.Reduction.MEAN)

        project_loss = project_loss_image + 0.3 * project_loss_pca

        # refine_loss = tf.losses.mean_squared_error(
        refine_loss = tf.losses.absolute_difference(predictions=gcn_texture, labels=pca_texture,
                                                    reduction=tf.losses.Reduction.MEAN)

        perception_loss = 1 - tf.reduce_mean(utils.cosine(input_feat, gcn_feat))

        var_losses = []
        gcn_skin_texture = tf.gather(gcn_texture, self.bfm.skin_index, axis=1)
        for i in range(3):
          _, variance = tf.nn.moments(gcn_skin_texture[..., i], axes=1)
          var_losses.append(variance)
        var_loss = tf.reduce_mean(var_losses)

        sym_diff = tf.gather(gcn_texture, self.bfm.left_index, axis=1) - tf.gather(
            gcn_texture, self.bfm.right_index, axis=1)
        sym_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(sym_diff) + 1e-16, axis=-1)))

        # adj_tensor = tf.constant(self.adjacent.reshape(
        #     [1, self.num_vert, self.num_vert, 1]),
        #                          dtype=tf.int32,
        #                          shape=[1, self.num_vert, self.num_vert, 1])
        # coo = self.adjacent.tocoo()

        # indices = np.mat([0, self.adjacent.row, self.adjacent.col, 0]).transpose()
        # values = np.ones_like(self.adjacent.data, np.float32)
        # adj_tensor = tf.SparseTensor(indices, values, self.adjacent.shape)
        # # adj_tensor = tf.SparseTensor(self.adjacent.indices,
        # #                             np.clip(self.adjacent.data, 0, 1),
        # #                             self.adjacent.shape)
        # expand = tf.ones([1, self.num_vert, self.num_vert, 3], dtype=tf.float32)
        # expand = expand * tf.expand_dims(gcn_texture, axis=1)
        # exp_trans = tf.transpose(expand, [0, 2, 1, 3])
        # # vertical = tf.ones([self.num_vert, self.num_vert, 3], dtype=tf.float32)
        # # vertical = vertical * tf.expand_dims(gcn_texture, axis=2)
        # smooth_loss = tf.abs((expand - exp_trans) * adj_tensor)
        # test = tf.sparse_to_dense(smooth_loss.indices, )

        #TODO: need attention
        # data_loss = self.ph_ref_lambda * refine_loss + self.ph_ren_lambda * (
        #     gcn_render_loss + 0.2 * project_loss +
        #     0.2 * perception_loss) + 0.1 * sym_loss
        data_loss = self.ph_ref_lambda * refine_loss + self.ph_ren_lambda * (
            project_loss + 0.2 * perception_loss + 0.5 * sym_loss + 0.01 * var_loss)

        # if not get_inter:
        #   self.skin_mask = skin_mask
        #   self.gcn_render_mask = gcn_render_mask
        #   self.gcn_render_image = gcn_render[..., :3]
        #   self.input_image_rgb = input_image[..., :3]
        #   self.pca_render_image = pca_render[..., :3]

      with tf.name_scope('regularization'):
        regularization *= tf.add_n(self.regularizers)
      loss = data_loss + regularization

      tf.summary.scalar('loss/data_loss', data_loss)
      tf.summary.scalar('loss/pca_render_loss', pca_render_loss)
      tf.summary.scalar('loss/gcn_render_loss', gcn_render_loss)
      tf.summary.scalar('loss/project_loss', project_loss)
      tf.summary.scalar('loss/refine_loss', refine_loss)
      tf.summary.scalar('loss/perception_loss', perception_loss)
      tf.summary.scalar('loss/var_loss', var_loss)
      tf.summary.scalar('loss/sym_loss', sym_loss)
      tf.summary.scalar('loss/regularization', regularization)

      logger.info('Successfully Computed Losses')

      return loss, pca_render_loss, gcn_render_loss, project_loss, refine_loss, perception_loss, var_loss, sym_loss

  def compute_gan_loss(self, real_image, fake_image, reuse=False, scale=10.0):
    t = not reuse
    real_score = self.image_disc(real_image, t, reuse=reuse)
    fake_score = self.image_disc(fake_image, t, reuse=True)

    epsilon = tf.random_uniform([], 0.0, 1.0)
    hat_image = epsilon * real_image + (1 - epsilon) * fake_image
    hat_score = self.image_disc(hat_image, t, reuse=True)
    hat_gradient = tf.gradients(hat_score, hat_image)[0]
    hat_gradient = tf.sqrt(tf.reduce_sum(tf.square(hat_gradient), axis=[1, 2, 3]))
    hat_gradient = tf.reduce_mean(tf.square(hat_gradient - 1.0) * scale)

    g_loss = -self.adv_lambda * tf.reduce_mean(fake_score)
    d_loss = self.adv_lambda * (tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) +
                                hat_gradient)

    logger.info('Successfully Computed GAN Losses')

    return g_loss, d_loss

  def training(self, g_loss, d_loss=None, decay_rate=0.98):
    """Adds to the loss model the Ops required to generate and apply gradients."""
    with tf.name_scope('training'):
      # Learning rate.
      global_step = tf.Variable(0, name='global_step', trainable=False)
      if decay_rate != 1:
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                   self.decay_steps, decay_rate, staircase=True)
      else:
        learning_rate = self.learning_rate

      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

      check_grads = []

      def check_gradients(grads):
        for i, (grad, var) in enumerate(grads):
          if grad is None:
            logger.info('warning: %s has no gradient', var.op.name)
          else:
            grads[i] = (tf.clip_by_norm(grad, 5), var)
            check_grads.append(tf.check_numerics(grad, "error occur"))

      all_vars = tf.trainable_variables()
      mesh_gen_vars = [x for x in all_vars if x.name.startswith('mesh_generator')]
      g_grads = optimizer.compute_gradients(g_loss, var_list=mesh_gen_vars)
      check_gradients(g_grads)

      if d_loss is not None:
        image_dis_vars = [x for x in all_vars if x.name.startswith('image_disc')]
        d_grads = optimizer.compute_gradients(d_loss, var_list=image_dis_vars)
        check_gradients(d_grads)

      with tf.control_dependencies(check_grads):
        op_g_grad = optimizer.apply_gradients(g_grads, global_step=global_step)
        if d_loss is not None:
          op_d_grad = optimizer.apply_gradients(d_grads, global_step=global_step)

      # The op return the learning rate.
      update_bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies([op_g_grad] + update_bn_ops):
        gen_train = tf.identity(learning_rate, name='control')

      dis_train = None
      if d_loss is not None:
        with tf.control_dependencies([op_d_grad] + update_bn_ops):
          dis_train = tf.identity(learning_rate, name='control')

      logger.info('Successfully Build Training Optimizer')

      return gen_train, dis_train

  def fit(self):
    for d in [self.ckpt_dir, self.summ_dir, self.samp_dir]:
      if not os.path.isdir(d):
        os.makedirs(d)

    logger.info('Start Fitting Model')
    t_process, t_wall = time.clock(), time.time()
    shutil.rmtree(self.summ_dir, ignore_errors=True)
    writer = tf.summary.FileWriter(self.summ_dir)
    # shutil.rmtree(self.ckpt_dir, ignore_errors=True)
    if not os.path.isdir(self.ckpt_dir):
      os.makedirs(self.ckpt_dir)
    path = os.path.join(self.ckpt_dir, 'model')
    if not os.path.isdir(self.samp_dir):
      os.makedirs(self.samp_dir)
    self.sess.run(self.op_init)

    if self.restore:
      self._restore_ckpt()
      self.restore = False

    val_image = utils.load_images(self.val_image_paths, self.img_size, alpha=True, landmark=False)

    step = 0
    for epoch in range(self.num_epochs):
      ren_lambda = np.clip(0.2 * epoch, 0, 1).astype(np.float32)
      ref_lambda = np.clip(1 - ren_lambda, 0.2, 1).astype(np.float32)
      logger.info('render_lambda: %f, refine_lambda: %f', ren_lambda, ref_lambda)
      feed_dict = {self.ph_ren_lambda: ren_lambda, self.ph_ref_lambda: ref_lambda}
      fetches = [
          self.gen_train, self.all_loss, self.pca_loss, self.gcn_loss, self.proj_loss,
          self.refine_loss, self.perc_loss, self.var_loss, self.sym_loss
      ]
      if self.gan:
        dis_fetches = fetches + [self.g_loss, self.d_loss]
      for batch in range(self.num_batches):
        try:
          train_dis = self.gan and ren_lambda > 1e-5
          # train_dis = True
          if train_dis:
            for _ in range(5):
              _ = self.sess.run(self.dis_train, feed_dict=feed_dict)

            _, all_loss, pca_loss, gcn_loss, proj_loss, refine_loss, perc_loss, var_loss, sym_loss, g_loss, d_loss = self.sess.run(
                dis_fetches, feed_dict=feed_dict)
          else:
            _, all_loss, pca_loss, gcn_loss, proj_loss, refine_loss, perc_loss, var_loss, sym_loss = self.sess.run(
                fetches, feed_dict=feed_dict)
          if batch % 10 == 0:
            log_str = ' all_loss: {:.3e}, pca_loss: {:.3e}, gcn_loss: {:.3e}, proj_loss: {:.3e}, refine_loss: {:.3e}, perc_loss: {:.3e}, var_loss: {:.3e}, sym_loss: {:.3e}'.format(
                all_loss, pca_loss, gcn_loss, proj_loss, refine_loss, perc_loss, var_loss, sym_loss)
            if train_dis:
              log_str += ', g_loss: {:.3e}, d_loss: {:.3e}'.format(g_loss, d_loss)
            logger.info('batch {} / {} (epoch {} / {}):'.format(batch, self.num_batches, epoch,
                                                                self.num_epochs))
            logger.info(log_str)
        except Exception as e:
          logger.info('Error Occured in Sess Run.')
          logger.debug(e)

        # Periodical evaluation of the model.
        if batch % self.eval_frequency == 0:
          string, results = self.evaluate(val_image)
          logger.info('  validation {}'.format(string))
          logger.info('  time: {:.0f}s (wall {:.0f}s)'.format(time.clock() - t_process,
                                                              time.time() - t_wall))
          self.save_sample(results, step, val_image, idx=0)

          # Summaries for TensorBoard.
          summary = tf.Summary(
              value=[tf.Summary.Value(tag='validation/loss', simple_value=results['all_loss'])])
          writer.add_summary(summary, step)

          # Save model parameters (for evaluation).
          self.op_saver.save(self.sess, path, global_step=step)
        step += 1

    writer.close()

  def save_sample(self, results, step, val_image, val_landmark=None, sample_dir=None, idx=0,
                  only_skin=False):
    if sample_dir is None:
      sample_dir = self.samp_dir

    input_image = utils.img_denormalize(val_image[idx])
    vertice = results['vertices'][idx]
    normal = results['normals'][idx]
    pca_texture = utils.img_denormalize(results['pca_texts'][idx])
    gcn_texture = utils.img_denormalize(results['gcn_texts'][idx])
    pca_color = utils.img_denormalize(results['pca_colors'][idx])
    gcn_color = utils.img_denormalize(results['gcn_colors'][idx])
    proj_color = utils.img_denormalize(results['proj_color'][idx])
    pca_ren_tex = utils.img_denormalize(results['pca_ren_texs'][idx])
    gcn_ren_tex = utils.img_denormalize(results['gcn_ren_texs'][idx])
    pca_ren_clr = utils.img_denormalize(results['pca_ren_clrs'][idx])
    gcn_ren_clr = utils.img_denormalize(results['gcn_ren_clrs'][idx])
    lm_proj = results['lm_projs'][idx]

    # input_image = np.clip(
    #     input_image.astype(np.int32) + [[[0, 0, 0, 64]]], 0,
    #     255).astype(np.uint8)
    imageio.imsave(os.path.join(sample_dir, '{}_input.png'.format(step)), input_image[..., :3])
    # imageio.imsave(os.path.join(sample_dir, '{}_mask.png'.format(step)),
    #                input_image[..., 3])
    if val_landmark is None:
      lm_image = input_image[..., :3]
    else:
      lm_image = utils.draw_image_with_lm(None, input_image[..., :3], val_landmark[idx],
                                          self.img_size, (0, 0, 255))
    utils.draw_image_with_lm(os.path.join(sample_dir, '{}_lm_proj.png'.format(step)), lm_image,
                             lm_proj, self.img_size)

    render_mask = pca_ren_clr[:, :, 3:] // 255
    if only_skin:
      render_mask = render_mask * (input_image[..., 3:] // 255)
      # render_mask = cv2.erode(render_mask, np.ones((5, 5), dtype=np.uint8), iterations=5)

    imageio.imsave(os.path.join(sample_dir, '{}_mask.png'.format(step)), render_mask * 255)

    def save_render(inputs, name, draw_lm=False):
      image = inputs[:, :, :3] * render_mask + input_image[:, :, :3] * (1 - render_mask)
      if draw_lm:
        utils.draw_image_with_lm(os.path.join(sample_dir, name), image, lm_proj, self.img_size)
      else:
        imageio.imsave(os.path.join(sample_dir, name), image)

    # imageio.imsave(os.path.join(sample_dir, '{}_gcn.png'.format(step)), gcn_ren_clr)
    save_render(pca_ren_tex, '{}_pca_ren_tex.png'.format(step))
    save_render(gcn_ren_tex, '{}_gcn_ren_tex.png'.format(step))
    save_render(pca_ren_clr, '{}_pca_ren_clr.png'.format(step))
    save_render(gcn_ren_clr, '{}_gcn_ren_clr.png'.format(step))

    write_obj(os.path.join(sample_dir, '{}_pca_texture.obj'.format(step)), vertice,
              self.refer_meshes[0]['faces'], pca_texture, normal)
    write_obj(os.path.join(sample_dir, '{}_gcn_texture.obj'.format(step)), vertice,
              self.refer_meshes[0]['faces'], gcn_texture, normal)
    write_obj(os.path.join(sample_dir, '{}_pca_color.obj'.format(step)), vertice,
              self.refer_meshes[0]['faces'], pca_color, normal)
    write_obj(os.path.join(sample_dir, '{}_gcn_color.obj'.format(step)), vertice,
              self.refer_meshes[0]['faces'], gcn_color, normal)
    write_obj(os.path.join(sample_dir, '{}_proj_color.obj'.format(step)), vertice,
              self.refer_meshes[0]['faces'], proj_color, normal)
    logger.info('Sample %s saved!', step)

  def evaluate(self, images):
    # t_process, t_wall = time.clock(), time.time()

    size = images.shape[0]
    result_list = []

    for begin in range(0, size, self.batch_size):
      end = begin + self.batch_size
      end = min([end, size])
      batch_image = np.zeros((self.batch_size, images.shape[1], images.shape[2], images.shape[3]))
      tmp_image = images[begin:end]
      batch_image[:end - begin] = tmp_image
      # batch_landmark = None
      # if landmarks is not None:
      #   batch_landmark = np.zeros((self.batch_size, len(self.lm_3d_idx), 2))
      #   tmp_landmark = landmarks[begin:end]
      #   batch_landmark[:end - begin] = tmp_landmark

      result = self.predict(batch_image)
      result_list.append(result)

    results = {
        'vertices': np.concatenate([x['vertice'] for x in result_list]),
        'normals': np.concatenate([x['normal'] for x in result_list]),
        'pca_texts': np.concatenate([x['pca_text'] for x in result_list]),
        'gcn_texts': np.concatenate([x['gcn_text'] for x in result_list]),
        'pca_colors': np.concatenate([x['pca_color'] for x in result_list]),
        'gcn_colors': np.concatenate([x['gcn_color'] for x in result_list]),
        'proj_color': np.concatenate([x['proj_color'] for x in result_list]),
        'pca_ren_texs': np.concatenate([x['pca_ren_tex'] for x in result_list]),
        'gcn_ren_texs': np.concatenate([x['gcn_ren_tex'] for x in result_list]),
        'pca_ren_clrs': np.concatenate([x['pca_ren_clr'] for x in result_list]),
        'gcn_ren_clrs': np.concatenate([x['gcn_ren_clr'] for x in result_list]),
        'lm_projs': np.concatenate([x['lm_proj'] for x in result_list]),
        'all_loss': np.mean([x['all_loss'] for x in result_list]),
        'pca_loss': np.mean([x['pca_loss'] for x in result_list]),
        'gcn_loss': np.mean([x['gcn_loss'] for x in result_list]),
        'proj_loss': np.mean([x['proj_loss'] for x in result_list]),
        'refine_loss': np.mean([x['refine_loss'] for x in result_list]),
        'perc_loss': np.mean([x['perc_loss'] for x in result_list]),
    }

    string = 'loss: {:.3e}, pca_loss:{:.3e}, gcn_loss:{:.3e}, proj_loss:{:.3e}, refine_loss:{:.3e}, perc_loss:{:.3e}'.format(
        result['all_loss'], result['pca_loss'], result['gcn_loss'], result['proj_loss'],
        result['refine_loss'], result['perc_loss'])

    if self.gan:
      results['g_loss'] = np.mean([x['g_loss'] for x in result_list])
      results['d_loss'] = np.mean([x['d_loss'] for x in result_list])
      string += ', g_loss:{:.3e}, d_loss:{:.3e}'.format(results['g_loss'], results['d_loss'])
    return string, results

  def predict(self, images):
    if not isinstance(images, np.ndarray):
      images = np.array(images)

    if self.restore:
      self._restore_ckpt()
      self.restore = False

    fetches = [
        self.vert_test, self.norm_test, self.pca_text_test, self.gcn_text_test, self.pca_color_test,
        self.gcn_color_test, self.proj_color_test, self.pca_ren_tex_test, self.gcn_ren_tex_test,
        self.pca_ren_clr_test, self.gcn_ren_clr_test, self.lm_proj_test, self.test_all_loss,
        self.test_pca_loss, self.test_gcn_loss, self.test_proj_loss, self.test_refine_loss,
        self.test_perc_loss
    ]
    feed_dict = {
        self.ph_rgbas: images,
        # self.ph_2dlms: landmarks,
        self.ph_ren_lambda: 1,
        self.ph_ref_lambda: 1
    }

    # coeff, feat, emb, resize = self.sess.run([
    #     self.coeff_test, self.image_feat_test, self.image_emb_test,
    #     self.resize_input
    # ],
    #                                          feed_dict=feed_dict)
    # imageio.imwrite('test1.png', resize[0].astype(np.uint8))

    if self.gan:
      fetches += [self.test_g_loss, self.test_d_loss]
      vertice, normal, pca_text, gcn_text, pca_color, gcn_color, proj_color, pca_ren_tex,\
      gcn_ren_tex, pca_ren_clr, gcn_ren_clr, lm_proj, all_loss, pca_loss, gcn_loss,\
      proj_loss, refine_loss, perc_loss, g_loss, d_loss = self.sess.run(
          fetches, feed_dict)
    else:
      vertice, normal, pca_text, gcn_text, pca_color, gcn_color, proj_color, pca_ren_tex,\
      gcn_ren_tex, pca_ren_clr, gcn_ren_clr, lm_proj, all_loss, pca_loss, gcn_loss,\
      proj_loss, refine_loss, perc_loss = self.sess.run(fetches, feed_dict)

    result = {
        'vertice': vertice,
        'normal': normal,
        'pca_text': pca_text,
        'gcn_text': gcn_text,
        'pca_color': pca_color,
        'gcn_color': gcn_color,
        'proj_color': proj_color,
        'pca_ren_tex': pca_ren_tex,
        'gcn_ren_tex': gcn_ren_tex,
        'pca_ren_clr': pca_ren_clr,
        'gcn_ren_clr': gcn_ren_clr,
        'lm_proj': lm_proj,
        'all_loss': all_loss,
        'pca_loss': pca_loss,
        'gcn_loss': gcn_loss,
        'proj_loss': proj_loss,
        'refine_loss': refine_loss,
        'perc_loss': perc_loss
    }

    if self.gan:
      result['g_loss'] = g_loss
      result['d_loss'] = d_loss

    return result

  def load_image(self, filename):
    return utils.load_image(filename, self.img_size, True, True)

  def load_image_bin(self, start_idx):
    image_len = 4 * self.img_size * self.img_size * 4
    return utils.load_image_bin(start_idx, self.img_file, image_len, self.img_size)

  def _erosion2d(self, inputs):
    # outputs = inputs
    outputs = tf.nn.dilation2d(inputs, self.dilation_kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME') - 1
    for _ in range(2):
      outputs = tf.nn.erosion2d(outputs, self.erosion_kernel, [1, 1, 1, 1], [1, 1, 1, 1],
                                'SAME') + 1
    return outputs

  def _restore_ckpt(self):
    # if self.serv_restore:
    #   filename = tf.train.latest_checkpoint(
    #       os.path.join(self.root_dir, self.ckpt_dir))
    # else:
    filename = tf.train.latest_checkpoint(self.ckpt_dir)
    if filename:
      self.op_saver.restore(self.sess, filename)
      logger.info('======================================')
      logger.info('Restored checkpoint from %s', filename)
      logger.info('======================================')

  def _weight_variable(self, shape, regularization=True, initial=None):
    # initial = tf.truncated_normal_initializer(0, 0.1)
    if initial is None:
      try:
        initial = tf.initializers.he_normal()
      except AttributeError:
        initial = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
    if regularization:
      self.regularizers.append(tf.nn.l2_loss(var))
    # tf.summary.histogram(var.op.name, var)
    return var

  def _bias_variable(self, shape, regularization=True, initial=tf.zeros_initializer()):
    #  initial=tf.constant_initializer(0.1)):
    var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
    if regularization:
      self.regularizers.append(tf.nn.l2_loss(var))
    # tf.summary.histogram(var.op.name, var)
    return var

  def chebyshev5(self, inputs, L, Fout, K):
    # if not hasattr(self, 'InterX'):
    # self.InterX = x
    N, M, Fin = inputs.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
    L = scipy.sparse.csr_matrix(L)
    L = graph.rescale_L(L, 2)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)
    # Transform to Chebyshev basis
    x0 = tf.transpose(inputs, perm=[1, 2, 0])  # M x Fin x N
    x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
    x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

    def concat(x, x_):
      x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
      return tf.concat([x, x_], axis=0)  # K x M x Fin*N

    if K > 1:
      x1 = tf.sparse_tensor_dense_matmul(L, x0)
      x = concat(x, x1)
    for _ in range(2, K):
      x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
      x = concat(x, x2)
      x0, x1 = x1, x2
    x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
    x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
    x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
    # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
    W = self._weight_variable([Fin * K, Fout], regularization=False)
    x = tf.matmul(x, W)  # N*M x Fout
    return tf.reshape(x, [N, M, Fout])  # N x M x Fout

  def cheb_res_block(self, inputs, L, Fout, K, relu=True):
    _, _, Fin = inputs.get_shape().as_list()
    if Fin != Fout:
      with tf.variable_scope('shortcut'):
        shortcut = self.chebyshev5(inputs, L, Fout, 1)
    else:
      shortcut = inputs

    with tf.variable_scope('filter1'):
      x = self.chebyshev5(inputs, L, Fout, K)
    with tf.variable_scope('bias_relu1'):
      x = self.brelu(x)

    with tf.variable_scope('filter2'):
      x = self.chebyshev5(x, L, Fout, K)
    x = tf.add(x, shortcut)
    if relu:
      with tf.variable_scope('bias_relu2'):
        x = self.brelu(x)

    # with tf.variable_scope('filter3'):
    #   x = self.chebyshev5(x, L, 3, K)
    # if tanh:
    #   x = tf.nn.tanh(x)

    return x

  def b1relu(self, inputs):
    """Bias and ReLU. One bias per filter."""
    # N, M, F = x.get_shape()
    _, _, F = inputs.get_shape()
    b = self._bias_variable([1, 1, int(F)], regularization=False)
    #TODO replace with tf.nn.elu
    # return tf.nn.relu(inputs + b)
    return tf.nn.elu(inputs + b)

  def b2relu(self, inputs):
    """Bias and ReLU. One bias per vertex per filter."""
    # N, M, F = x.get_shape()
    _, M, F = inputs.get_shape()
    b = self._bias_variable([1, int(M), int(F)], regularization=False)
    return tf.nn.relu(inputs + b)

  def poolwT(self, inputs, L):
    Mp = L.shape[0]
    N, M, Fin = inputs.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
    L = scipy.sparse.csr_matrix(L)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)

    x = tf.transpose(inputs, perm=[1, 2, 0])  # M x Fin x N
    x = tf.reshape(x, [M, Fin * N])  # M x Fin*N
    x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
    x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
    x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin

    return x

  def fc(self, inputs, Mout, relu=True):
    """Fully connected layer with Mout features."""
    # N, Min = x.get_shape()
    _, Min = inputs.get_shape()
    W = self._weight_variable([int(Min), Mout], regularization=True)
    b = self._bias_variable([Mout], regularization=True)
    x = tf.matmul(inputs, W) + b
    return tf.nn.relu(x) if relu else x

  def conv2d(self, inputs, f_out, kernel, stride, padding='SAME', batch_norm=True, lrelu=True,
             is_training=True, name='conv2d'):
    with tf.variable_scope(name):
      _, _, _, f_in = inputs.get_shape()
      W = self._weight_variable([kernel, kernel, f_in, f_out])
      b = self._bias_variable([f_out])
      # b = self._bias_variable([1, 28, 28, self.F])
      # x_2d = tf.reshape(x, [-1,28,28,1])
      x = tf.nn.conv2d(inputs, W, strides=[1, stride, stride, 1], padding=padding) + b
      if batch_norm:
        x = tf.contrib.layers.batch_norm(x, decay=0.9, zero_debias_moving_mean=True,
                                         is_training=is_training, trainable=True)
      return tf.nn.leaky_relu(x) if lrelu else x

  def compute_norm(self, vertice):
    # vertex index for each triangle face, with shape [F,3], F is number of faces
    face_id = self.refer_faces[0]
    # adjacent face index for each vertex, with shape [N,8], N is number of vertex
    point_id = self.bfm.point_buf - 1
    point_id = tf.reshape(point_id, [-1])
    v1 = tf.gather(vertice, face_id[:, 0], axis=1)
    v2 = tf.gather(vertice, face_id[:, 1], axis=1)
    v3 = tf.gather(vertice, face_id[:, 2], axis=1)
    e1 = v1 - v2
    e2 = v2 - v3

    face_norm = tf.cross(e1, e2)  # compute normal for each face
    # concat face_normal with a zero vector at the end
    face_norm = tf.concat([face_norm, tf.zeros([self.batch_size, 1, 3], dtype=tf.float32)], axis=1)
    v_norms = tf.gather(face_norm, point_id, axis=1)
    # compute vertex normal using one-ring neighborhood
    v_norm = tf.reduce_sum(tf.reshape(v_norms, [self.batch_size, self.num_vert, 8, 3]), axis=2)
    # normalize normal vectors
    v_norm = v_norm / tf.expand_dims(tf.linalg.norm(v_norm, axis=2), 2)

    return v_norm

  def project_color(self, proj_vert, image):
    half_size = self.img_size // 2
    vertice = tf.cast(tf.round(proj_vert * half_size + half_size), np.int32)
    flatten_image = tf.reshape(image, [self.batch_size, self.img_size * self.img_size, -1])
    x_coords = tf.clip_by_value(vertice[..., 1], 0, self.img_size - 1)
    y_coords = tf.clip_by_value(vertice[..., 0], 0, self.img_size - 1)
    coords = x_coords * self.img_size + y_coords
    # proj_color = tf.gather_nd(flatten_image, coords)
    proj_color = utils.batch_gather(flatten_image, coords)
    return proj_color

  def illumination(self, face_tex, norm, gamma):
    # input face_tex should be [0, 1] with RGB channels
    face_texture = face_tex * 255.0

    init_lit = tf.constant([0.8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
    gamma = tf.reshape(gamma, [-1, 3, 9])
    gamma = gamma + tf.reshape(init_lit, [1, 1, 9])

    a0 = np.pi
    a1 = 2 * np.pi / tf.sqrt(3.0)
    a2 = 2 * np.pi / tf.sqrt(8.0)
    c0 = 1 / tf.sqrt(4 * np.pi)
    c1 = tf.sqrt(3.0) / tf.sqrt(4 * np.pi)
    c2 = 3 * tf.sqrt(5.0) / tf.sqrt(12 * np.pi)

    Y_shape = [self.batch_size, self.num_vert, 1]
    Y0 = tf.tile(tf.reshape(a0 * c0, [1, 1, 1]), Y_shape)
    Y1 = tf.reshape(-a1 * c1 * norm[:, :, 1], Y_shape)
    Y2 = tf.reshape(a1 * c1 * norm[:, :, 2], Y_shape)
    Y3 = tf.reshape(-a1 * c1 * norm[:, :, 0], Y_shape)
    Y4 = tf.reshape(a2 * c2 * norm[:, :, 0] * norm[:, :, 1], Y_shape)
    Y5 = tf.reshape(-a2 * c2 * norm[:, :, 1] * norm[:, :, 2], Y_shape)
    Y6 = tf.reshape(a2 * c2 * 0.5 / tf.sqrt(3.0) * (3 * tf.square(norm[:, :, 2]) - 1), Y_shape)
    Y7 = tf.reshape(-a2 * c2 * norm[:, :, 0] * norm[:, :, 2], Y_shape)
    Y8 = tf.reshape(a2 * c2 * 0.5 * (tf.square(norm[:, :, 0]) - tf.square(norm[:, :, 1])), Y_shape)
    Y = tf.concat([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], axis=2)

    lit_r = tf.squeeze(tf.matmul(Y, tf.expand_dims(gamma[:, 0, :], 2)), 2)
    # [batch,N,9] * [batch,9,1] = [batch,N]
    lit_g = tf.squeeze(tf.matmul(Y, tf.expand_dims(gamma[:, 1, :], 2)), 2)
    lit_b = tf.squeeze(tf.matmul(Y, tf.expand_dims(gamma[:, 2, :], 2)), 2)

    face_color = tf.stack([
        lit_r * face_texture[:, :, 0], lit_g * face_texture[:, :, 1], lit_b * face_texture[:, :, 2]
    ], axis=2)
    # lighting = np.stack([lit_r, lit_g, lit_b], axis=2) * 128

    return tf.clip_by_value(face_color / 255.0, 0.0, 1.0)

  def mesh_generator(self, *args, **kwargs):
    raise NotImplementedError()

  def image_disc(self, *args, **kwargs):
    raise NotImplementedError()
