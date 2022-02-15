import logging
import os
import random
from glob import glob

import cv2
import h5py
import imageio
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import tensorflow as tf
from PIL import Image

from lib import graph, mesh_sampling, spatialsearch
from lib.mesh_io import read_obj


def make_paths(filenames, prefix, root_dir):
  train_txt = '{}_train.txt'.format(prefix)
  val_txt = '{}_val.txt'.format(prefix)
  test_txt = '{}_test.txt'.format(prefix)
  if os.path.isfile(test_txt):
    with open(train_txt, 'r') as f:
      train_paths = [os.path.join(root_dir, p.strip()) for p in f.readlines()]
    with open(val_txt, 'r') as f:
      val_paths = [os.path.join(root_dir, p.strip()) for p in f.readlines()]
    with open(test_txt, 'r') as f:
      test_paths = [os.path.join(root_dir, p.strip()) for p in f.readlines()]
  else:
    if not os.path.isdir(os.path.split(prefix)[0]):
      os.makedirs(os.path.split(prefix)[0])
    if 'image' in prefix:

      if not 'mv' in prefix:

        def check_lm(im_f):
          lm_f = im_f.replace('_Segment', '_Landmarks')
          lm_f = lm_f.replace('jpg', 'npy')
          lm_f = lm_f.replace('png', 'npy')
          return os.path.isfile(lm_f)

        filenames = [f for f in filenames if check_lm(f)]

    train_num = len(filenames) - 8192
    random.shuffle(filenames)
    train_paths = filenames[:train_num]
    val_paths = filenames[train_num:-4096]
    test_paths = filenames[-4096:]
    with open(train_txt, 'w') as f:
      f.writelines(p.split(root_dir, 1)[1][1:] + '\n' for p in train_paths)
    with open(val_txt, 'w') as f:
      f.writelines(p.split(root_dir, 1)[1][1:] + '\n' for p in val_paths)
    with open(test_txt, 'w') as f:
      f.writelines(p.split(root_dir, 1)[1][1:] + '\n' for p in test_paths)
  return train_paths, val_paths, test_paths


def load_image_bin(start_idx, img_file, image_len, img_size):
  img_start = start_idx * image_len
  # lm_start = start_idx * landmark_len

  img_file.seek(img_start)
  im_str = img_file.read(image_len)
  # lm_file.seek(lm_start)
  # lm_str = lm_file.read(landmark_len)

  images = np.fromstring(im_str, dtype=np.float32).reshape((img_size, img_size, 4))
  # landmarks = np.fromstring(lm_str, dtype=np.float32).reshape((51, 2))
  # return assets, landmarks
  return images


class ImageCropper():

  def __init__(self, predictor_path, img_size):
    import dlib
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(predictor_path)
    self.load_lm3d()
    self.img_size = img_size

  def load_lm3d(self):
    Lm3D = sio.loadmat('data/similarity_Lm3D_all.mat')
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([
        Lm3D[lm_idx[0], :],
        np.mean(Lm3D[lm_idx[[1, 2]], :], 0),
        np.mean(Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]
    ], axis=0)
    self.lm3D = Lm3D[[1, 2, 0, 3, 4], :]

  def compute_lm_trans(self, lm):
    npts = lm.shape[1]
    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = self.lm3D
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = self.lm3D
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(lm.transpose(), [2 * npts, 1])
    k, _, _, _ = np.linalg.lstsq(A, b, -1)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

  def process_image(self, img, lm, t, s):
    w0, h0 = img.size
    img = img.transform(img.size, Image.AFFINE, (1, 0, t[0] - w0 / 2, 0, 1, h0 / 2 - t[1]))

    half_size = self.img_size // 2
    # scale = half_size - 10 * (self.img_size / 224)
    scale = (102 / 224) * self.img_size

    # w = (w0 / s * 102).astype(np.int32)
    # h = (h0 / s * 102).astype(np.int32)
    w = (w0 / s * scale).astype(np.int32)
    h = (h0 / s * scale).astype(np.int32)
    img = img.resize((w, h), resample=Image.BILINEAR)
    # lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2],
    #               axis=1) / s * 102

    # crop the image to 224*224 from image center
    left = (w / 2 - half_size).astype(np.int32)
    right = left + self.img_size
    up = (h / 2 - half_size).astype(np.int32)
    below = up + self.img_size

    img = img.crop((left, up, right, below))
    img = np.array(img)
    # img = img[:, :, ::-1]
    # img = np.expand_dims(img, 0)
    # lm = lm - np.reshape(np.array([(w / 2 - half_size),
    #                                (h / 2 - half_size)]), [1, 2])

    return img

  def extend_img(self, inputs):
    width, height, _ = inputs.shape
    top = int(height * 0.3)
    left = int(width * 0.3)

    outputs = cv2.copyMakeBorder(inputs, top, top, left, left, cv2.BORDER_REPLICATE)
    return outputs

  def get_landmarks(self, image):
    faces = self.detector(np.array(image[..., :3]), 1)
    landmarks = self.predictor(np.array(image[..., :3]), faces[0])
    return landmarks

  def crop_image(self, image):
    image = self.extend_img(image)

    landmarks = self.get_landmarks(image)
    idxs = [[36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47], [30], [48], [54]]
    lm = np.zeros([5, 2])
    for i in range(5):
      for j in idxs[i]:
        lm[i] += np.array([landmarks.part(j).x, landmarks.part(j).y])
      lm[i] = lm[i] // len(idxs[i])

    new_image = Image.fromarray(image)
    w0, h0 = new_image.size

    lm = np.stack([lm[:, 0], h0 - 1 - lm[:, 1]], axis=1)
    t, s = self.compute_lm_trans(lm.transpose())

    return self.process_image(new_image, lm, t, s)


def load_image(filename, img_size, alpha, landmark, cropper=None, gray=False):
  if isinstance(filename, str):
    im_f = filename
  else:
    im_f = filename.decode()

  image = cv2.imread(im_f, cv2.IMREAD_UNCHANGED)
  image = cv2.resize(image, (img_size, img_size))
  if alpha:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    if cropper is not None:
      image = cropper.crop_image(image)
    image = image / [[[127.5, 127.5, 127.5, 255.0]]] - [[[1.0, 1.0, 1.0, 0.0]]]
  else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if cropper is not None:
      image = cropper.crop_image(image)
    if gray:
      image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image / 127.5 - 1.0

  if not landmark:
    return image.astype(np.float32)
  else:
    lm_f = im_f.replace('_Segment', '_Landmarks')
    lm_f = lm_f.replace('jpg', 'npy')
    lm_f = lm_f.replace('png', 'npy')
    landmarks = np.fromfile(lm_f, dtype=np.int32)
    landmarks = np.reshape(landmarks, [68, 2]).astype(np.float32)
    landmarks = landmarks[17:]
    half_size = img_size / 2
    landmarks = landmarks / half_size - 1.0
    return image.astype(np.float32), landmarks.astype(np.float32)


def load_images(filenames, img_size, alpha, landmark, cropper=None, gray=False):
  images = []
  if not landmark:
    for f in filenames:
      image = load_image(f, img_size, alpha, landmark, cropper, gray)
      images.append(image)
    return np.array(images)
  else:
    landmarks = []
    for f in filenames:
      image, lm = load_image(f, img_size, alpha, landmark, cropper, gray)
      images.append(image)
      landmarks.append(lm)
    return np.array(images), np.array(landmarks)


def load_mv_image(filedir, img_size, alpha, landmark):
  if isinstance(filedir, str):
    im_f = filedir
  else:
    im_f = filedir.decode()

  im_f0 = os.path.join(im_f, '0.png')
  im_f1s = glob('{}/[1-4].png'.format(im_f))
  # print(filedir, im_f0, im_f1s)
  im_f1 = random.choice(im_f1s)
  im_f2s = glob('{}/[5-8].png'.format(im_f))
  im_f2 = random.choice(im_f2s)

  loaded_data = load_images([im_f0, im_f1, im_f2], img_size, alpha, landmark)
  return loaded_data


def load_mv_images(filedirs, img_size, alpha, landmark):
  images = []
  if not landmark:
    for d in filedirs:
      image = load_mv_image(d, img_size, alpha, landmark)
      images.append(image)
    return np.array(images)
  else:
    landmarks = []
    for d in filedirs:
      image, lm = load_mv_image(d, img_size, alpha, landmark)
      images.append(image)
      landmarks.append(lm)
    return np.array(image), np.array(landmarks)


def init_sampling(refer_mesh, data_dir, dataname, ds_factors=(4, 4, 4, 4)):
  # Sampling factor of the mesh at each stage of sampling

  # Generates adjecency matrices A, downsampling matrices D, and upsamling matrices U by sampling
  # the mesh 4 times. Each time the mesh is sampled by a factor of 4
  adj_path = os.path.join(data_dir, 'adjacency')
  ds_path = os.path.join(data_dir, 'downsamp_trans')
  us_path = os.path.join(data_dir, 'upsamp_trans')
  lap_path = os.path.join(data_dir, 'laplacians')

  if not os.path.isfile(lap_path + '0.npz'):
    logger = logging.getLogger('x')
    logger.info('Computing Sampling Parameters')
    adjacencies, downsamp_trans, upsamp_trans = mesh_sampling.generate_transform_matrices(
        dataname, refer_mesh['vertices'], refer_mesh['faces'], ds_factors)
    adjacencies = [x.astype('float32') for x in adjacencies]
    downsamp_trans = [x.astype('float32') for x in downsamp_trans]
    upsamp_trans = [x.astype('float32') for x in upsamp_trans]
    laplacians = [graph.laplacian(a, normalized=True) for a in adjacencies]

    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    for i, a in enumerate(adjacencies):
      sp.save_npz(adj_path + '{}.npz'.format(i), a)
    for i, d in enumerate(downsamp_trans):
      sp.save_npz(ds_path + '{}.npz'.format(i), d)
    for i, u in enumerate(upsamp_trans):
      sp.save_npz(us_path + '{}.npz'.format(i), u)
    for i, l in enumerate(laplacians):
      sp.save_npz(lap_path + '{}.npz'.format(i), l)
  else:
    adjacencies = []
    downsamp_trans = []
    upsamp_trans = []
    laplacians = []
    for a in sorted(glob('{}*.npz'.format(adj_path))):
      adjacencies.append(sp.load_npz(a))
    for d in sorted(glob('{}*.npz'.format(ds_path))):
      downsamp_trans.append(sp.load_npz(d))
    for u in sorted(glob('{}*.npz'.format(us_path))):
      upsamp_trans.append(sp.load_npz(u))
    for l in sorted(glob('{}*.npz'.format(lap_path))):
      laplacians.append(sp.load_npz(l))

  pool_size = [x.shape[0] for x in adjacencies]
  return laplacians, downsamp_trans, upsamp_trans, pool_size


def split_bfm09_coeff(coeff):
  shape_coef = coeff[:, :80]  # identity(shape) coeff of dim 80
  exp_coef = coeff[:, 80:144]  # expression coeff of dim 64
  color_coef = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
  angles = coeff[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
  gamma = coeff[:, 227:254]  # lighting coeff for 3 channel SH function of dim 27
  translation = coeff[:, 254:]  # translation coeff of dim 3

  return shape_coef, exp_coef, color_coef, angles, gamma, translation


def get_mesh_list(name='bfm_face',):
  mesh_list = []
  for i in range(5):
    path = os.path.join('data', 'reference', name, 'reference{}.obj'.format(i))
    mesh_list.append(read_obj(path))

  return mesh_list


def image_augment(image, augment_size):
  seed = random.randint(0, 2**31 - 1)
  ori_image_shape = tf.shape(image)
  image = tf.image.random_flip_left_right(image, seed=seed)
  image = tf.image.resize_images(image, [augment_size, augment_size])
  image = tf.random_crop(image, ori_image_shape, seed=seed)
  return image


def img_normalize(images):
  return images / 127.5 - 1.0


def img_denormalize(image):
  images = np.clip(image, -1, 1)
  if np.shape(images)[-1] == 4:
    shape = [1] * (len(np.shape(images)) - 1) + [4]
    plus = np.reshape([1, 1, 1, 0], shape).astype(np.float32)
    mult = np.reshape([127.5, 127.5, 127.5, 255], shape).astype(np.float32)
    output = (images + plus) * mult
    # output = np.concatenate(
    #     [(assets[..., :3] + 1) * 127.5, assets[..., 3:] * 255.0], axis=-1)
  else:
    output = (images + 1) * 127.5
  return np.clip(output, 0, 255).astype(np.uint8)


def cosine(x, y):
  x_len = tf.sqrt(tf.reduce_sum(x * x, 1))
  y_len = tf.sqrt(tf.reduce_sum(y * y, 1))
  inner_product = tf.reduce_sum(x * y, 1)
  result = tf.div(inner_product, x_len * y_len + 1e-8, name='cosine_dist')
  return result


def cosine_np(x, y):
  x_len = np.sqrt(np.sum(x * x, 1))
  y_len = np.sqrt(np.sum(y * y, 1))
  inner_product = np.sum(x * y, 1)
  result = inner_product / (x_len * y_len + 1e-8)
  return result


def rotation_matrix_np(angles):
  angle_x = angles[:, 0]
  angle_y = angles[:, 1]
  angle_z = angles[:, 2]

  ones = np.ones_like(angle_x)
  zeros = np.zeros_like(angle_x)

  # yapf: disable
  rotation_X = np.array([[ones, zeros, zeros],
                         [zeros, np.cos(angle_x), -np.sin(angle_x)],
                         [zeros, np.sin(angle_x), np.cos(angle_x)]],
                        dtype=np.float32)
  rotation_Y = np.array([[np.cos(angle_y), zeros, np.sin(angle_y)],
                         [zeros, ones, zeros],
                         [-np.sin(angle_y), zeros, np.cos(angle_y)]],
                        dtype=np.float32)
  rotation_Z = np.array([[np.cos(angle_z), -np.sin(angle_z), zeros],
                         [np.sin(angle_z), np.cos(angle_z), zeros],
                         [zeros, zeros, ones]],
                        dtype=np.float32)
  # yapf: enable

  rotation_X = np.transpose(rotation_X, (2, 0, 1))
  rotation_Y = np.transpose(rotation_Y, (2, 0, 1))
  rotation_Z = np.transpose(rotation_Z, (2, 0, 1))
  rotation = np.matmul(np.matmul(rotation_Z, rotation_Y), rotation_X)
  # transpose row and column (dimension 1 and 2)
  rotation = np.transpose(rotation, axis=[0, 2, 1])

  return rotation


def rotation_matrix_tf(angles):
  angle_x = angles[:, 0]
  angle_y = angles[:, 1]
  angle_z = angles[:, 2]

  ones = tf.ones_like(angle_x)
  zeros = tf.zeros_like(angle_x)

  # yapf: disable
  rotation_X = tf.convert_to_tensor(
      [[ones, zeros, zeros],
       [zeros, tf.cos(angle_x), -tf.sin(angle_x)],
       [zeros, tf.sin(angle_x), tf.cos(angle_x)]],
      dtype=np.float32)
  rotation_Y = tf.convert_to_tensor(
      [[tf.cos(angle_y), zeros, tf.sin(angle_y)],
       [zeros, ones, zeros],
       [-tf.sin(angle_y), zeros, tf.cos(angle_y)]],
      dtype=tf.float32)
  rotation_Z = tf.convert_to_tensor(
      [[tf.cos(angle_z), -tf.sin(angle_z), zeros],
       [tf.sin(angle_z), tf.cos(angle_z), zeros],
       [zeros, zeros, ones]],
      dtype=tf.float32)
  # yapf: enable

  rotation_X = tf.transpose(rotation_X, (2, 0, 1))
  rotation_Y = tf.transpose(rotation_Y, (2, 0, 1))
  rotation_Z = tf.transpose(rotation_Z, (2, 0, 1))
  rotation = tf.matmul(tf.matmul(rotation_Z, rotation_Y), rotation_X)
  # transpose row and column (dimension 1 and 2)
  rotation = tf.transpose(rotation, perm=[0, 2, 1])

  return rotation


def illumination_np(face_texture, norm, gamma):

  num_vertex = np.shape(face_texture)[1]

  init_lit = np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0])
  gamma = np.reshape(gamma, [-1, 3, 9])
  gamma = gamma + np.reshape(init_lit, [1, 1, 9])

  # parameter of 9 SH function
  a0 = np.pi
  a1 = 2 * np.pi / np.sqrt(3.0)
  a2 = 2 * np.pi / np.sqrt(8.0)
  c0 = 1 / np.sqrt(4 * np.pi)
  c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
  c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)

  Y0 = np.tile(np.reshape(a0 * c0, [1, 1, 1]), [1, num_vertex, 1])
  Y1 = np.reshape(-a1 * c1 * norm[:, :, 1], [1, num_vertex, 1])
  Y2 = np.reshape(a1 * c1 * norm[:, :, 2], [1, num_vertex, 1])
  Y3 = np.reshape(-a1 * c1 * norm[:, :, 0], [1, num_vertex, 1])
  Y4 = np.reshape(a2 * c2 * norm[:, :, 0] * norm[:, :, 1], [1, num_vertex, 1])
  Y5 = np.reshape(-a2 * c2 * norm[:, :, 1] * norm[:, :, 2], [1, num_vertex, 1])
  Y6 = np.reshape(a2 * c2 * 0.5 / np.sqrt(3.0) * (3 * np.square(norm[:, :, 2]) - 1),
                  [1, num_vertex, 1])
  Y7 = np.reshape(-a2 * c2 * norm[:, :, 0] * norm[:, :, 2], [1, num_vertex, 1])
  Y8 = np.reshape(a2 * c2 * 0.5 * (np.square(norm[:, :, 0]) - np.square(norm[:, :, 1])),
                  [1, num_vertex, 1])
  Y = np.concatenate([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], axis=2)

  # Y shape:[batch,N,9].
  lit_r = np.squeeze(np.matmul(Y, np.expand_dims(gamma[:, 0, :], 2)),
                     2)  # [batch,N,9] * [batch,9,1] = [batch,N]
  lit_g = np.squeeze(np.matmul(Y, np.expand_dims(gamma[:, 1, :], 2)), 2)
  lit_b = np.squeeze(np.matmul(Y, np.expand_dims(gamma[:, 2, :], 2)), 2)

  # shape:[batch,N,3]
  face_color = np.stack(
      [lit_r * face_texture[:, :, 0], lit_g * face_texture[:, :, 1], lit_b * face_texture[:, :, 2]],
      axis=2)
  # lighting = np.stack([lit_r, lit_g, lit_b], axis=2) * 128

  return face_color


class LSFM_model(object):

  def __init__(self, root_dir, path='data/LSFM_boxer.mat'):
    super(LSFM_model, self).__init__()

    self.root_dir = root_dir
    self.path = os.path.join(root_dir, path)
    self.load_LSFM_boxer()
    self.compute_offset()

    self.n_shape_coef = self.shapePC.shape[1]
    self.n_exp_coef = self.expPC.shape[1]
    self.n_tex_coef = self.texPC.shape[1]
    self.n_all_coef = self.n_shape_coef + self.n_exp_coef + self.n_tex_coef

  def load_LSFM_boxer(self):
    C = sio.loadmat(self.path)
    model = C['model']
    model = model[0, 0]

    # change dtype from double(np.float64) to np.float32,
    # since big matrix process(espetially matrix dot) is too slow in python.
    self.shapeMU = model['shapeMU'].astype(np.float32)
    self.shapePC = model['shapePC'].astype(np.float32)
    self.expMU = model['expMU'].astype(np.float32)
    self.expPC = model['expPC'].astype(np.float32)
    self.texMU = model['texMU'].astype(np.float32)
    self.texPC = model['texPC'].astype(np.float32)

  def compute_offset(self):
    mean = read_obj(os.path.join('data/3dmd_mean.obj'))['vertices']
    lsfm = read_obj(os.path.join('data/lsfm_template.obj'))['vertices']
    idxs = [40502, 47965, 18958, 35610]
    scale = [(mean[idxs[0]] - mean[idxs[1]]) / (lsfm[idxs[0]] - lsfm[idxs[1]]),
             (mean[idxs[3]] - mean[idxs[2]]) / (lsfm[idxs[3]] - lsfm[idxs[2]])]

    self.scale = np.mean([scale[0][1], scale[1][0]])
    # scale = 1.17
    lsfm_scale = lsfm * self.scale
    self.offset = np.mean(mean, axis=0) - np.mean(lsfm_scale, axis=0)


class BFM_model(object):

  def __init__(self, root_dir, path):
    super(BFM_model, self).__init__()

    self.root_dir = root_dir
    self.path = os.path.join(root_dir, path)
    if '09' in path:
      self.load_BFM09()
    elif '17' in path:
      self.load_BFM17()

    self.n_shape_coef = self.shapePC.shape[1]
    self.n_exp_coef = self.expressionPC.shape[1]
    self.n_color_coef = self.colorPC.shape[1]
    self.n_all_coef = self.n_shape_coef + self.n_exp_coef + self.n_color_coef

  def load_BFM09(self):
    model = sio.loadmat(self.path)
    self.shapeMU = model['meanshape'].astype(np.float32)  # mean face shape
    self.shapePC = model['idBase'].astype(np.float32)  # identity basis
    self.expressionPC = model['exBase'].astype(np.float32)  # expression basis
    self.colorMU = model['meantex'].astype(np.float32)  # mean face texture
    self.colorPC = model['texBase'].astype(np.float32)  # texture basis
    self.point_buf = model['point_buf'].astype(np.int32)
    # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
    self.triangles = model['tri'].astype(np.int32)
    # vertex index for each triangle face, starts from 1
    self.landmark = np.squeeze(model['keypoints']).astype(
        np.int32) - 1  # 68 face landmark index, starts from 0
    skin_mask = sio.loadmat(os.path.join(self.root_dir,
                                         'data/bfm2009_4seg.mat'))['face05_4seg'][:, 0]
    face_id = sio.loadmat(os.path.join(self.root_dir,
                                       'data/bfm2009_face_idx.mat'))['select_id'][:, 0] - 1
    skin_mask = skin_mask[face_id]
    self.skin_index = np.where(skin_mask == 3)[0]

    sym_index = [
        x.split() for x in open('data/bfm2009_symlist.txt', 'r').readlines() if len(x.strip()) > 1
    ]
    sym_index = np.array([[int(x) for x in y] for y in sym_index])
    self.left_index = sym_index[:, 0]
    self.right_index = sym_index[:, 1]
    # crop_sym_idx = []
    # for x, y in sym_index:
    #   if x in face_id and y in face_id:
    #     crop_sym_idx.append([
    #         np.squeeze(np.where(face_id == x)),
    #         np.squeeze(np.where(face_id == y))
    #     ])

    # with open('data/bfm2009_face_symlist.txt', 'w') as f:
    #   for x, y in crop_sym_idx:
    #     f.write('{} {}\n'.format(x, y))

    # self.skin_mask = np.reshape(skin_mask == 3, [-1, 1]).astype(np.float32)
    # self.skin_mask = np.reshape(model['skinmask'].astype(np.int32), [-1, 1])
    # self.front_mask = np.zeros_like(self.skin_mask)
    # self.front_mask[np.reshape(model['frontmask2_idx'].astype(np.int32) - 1,
    #                            [-1, 1])] = 1

  def load_BFM17(self):
    with h5py.File(self.path, 'r') as hf:
      self.triangles = np.transpose(np.array(hf['shape/representer/cells']), [1, 0])

      self.shapeMU = np.array(hf['shape/model/mean']) / 1e2
      shape_orthogonal_pca_basis = np.array(hf['shape/model/pcaBasis'])
      shape_pca_variance = np.array(hf['shape/model/pcaVariance']) / 1e4

      self.colorMU = np.array(hf['color/model/mean'])
      color_orthogonal_pca_basis = np.array(hf['color/model/pcaBasis'])
      color_pca_variance = np.array(hf['color/model/pcaVariance'])

      self.expressionMU = np.array(hf['expression/model/mean']) / 1e2
      expression_pca_basis = np.array(hf['expression/model/pcaBasis'])
      expression_pca_variance = np.array(hf['expression/model/pcaVariance']) / 1e4

      self.shapePC = shape_orthogonal_pca_basis * np.expand_dims(np.sqrt(shape_pca_variance), 0)
      self.colorPC = color_orthogonal_pca_basis * np.expand_dims(np.sqrt(color_pca_variance), 0)
      self.expressionPC = expression_pca_basis * np.expand_dims(np.sqrt(expression_pca_variance), 0)


def get_vert_connectivity(mesh_v, mesh_f):
  """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

  vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

  # for each column in the faces...
  for i in range(3):
    IS = mesh_f[:, i]
    JS = mesh_f[:, (i + 1) % 3]
    data = np.ones(len(IS))
    # ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
    ij = np.vstack((IS.reshape((1, -1)), JS.reshape(1, -1)))
    mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
    vpv = vpv + mtx + mtx.T

  return vpv


def get_vertices_per_edge(mesh_v, mesh_f):
  """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

  vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
  # result = np.hstack((col(vc.row), col(vc.col)))
  result = np.hstack((vc.row.reshape(-1, 1), vc.col.reshape(-1, 1)))
  result = result[result[:, 0] < result[:, 1]]  # for uniqueness

  return result


def aabbtree_compute_nearest(src_vert, src_tri, tgt_vert, nearest_part=False):
  cpp_handle = spatialsearch.aabbtree_compute(
      np.array(src_vert).astype(np.float64).copy(order='C'),
      np.array(src_tri).astype(np.uint32).copy(order='C'))
  f_idxs, f_part, v = spatialsearch.aabbtree_nearest(
      cpp_handle, np.array(tgt_vert, dtype=np.float64, order='C'))
  return (f_idxs, f_part, v) if nearest_part else (f_idxs, v)


def init_logger(name='x', filename='log.txt'):
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
                                datefmt='%m-%d %H:%M:%S')

  fh = logging.FileHandler(filename, encoding='utf-8')
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  ch.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)

  return logger


def draw_image_with_lm(filename, inputs, landmarks, img_size, color=(0, 255, 0)):
  # image = image[:, :, :3]
  # image = image * 127.5 + 127.5
  # image = image.astype(np.uint8)
  # image = img_denormalize(image)
  image = inputs.copy()
  if np.max(landmarks) <= 1:
    half_size = img_size // 2
    landmarks = np.round(landmarks * half_size + half_size).astype(np.int32)
  # lm_2d[:, 1] = 224 - lm_2d[:, 1]
  for _, (x, y) in enumerate(landmarks):
    # try:
    # print(np.shape(image))
    # print(np.max(image))
    if np.shape(image)[-1] == 3:
      cv2.circle(image, (x, y), 1, color, -1, 8)
      # cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
      #             (255, 255, 255))
    else:
      cv2.circle(image, (x, y), 1, color + (255,), -1, 8)
      # cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
      #             (255, 255, 255, 255))
    # except IndexError as e:
    #   print(e)
  if filename is not None:
    imageio.imsave(filename, image)
  else:
    return image


def batch_gather(params, indices, name=None):
  """Gather slices from params according to indices with leading batch dims."""
  with tf.name_scope(name, "BatchGather", [params, indices]):
    indices = tf.convert_to_tensor(indices, name="indices")
    params = tf.convert_to_tensor(params, name="params")
    if indices.shape.ndims is None:
      raise ValueError("batch_gather does not allow indices with unknown shape.")
    return _batch_gather(params, indices, batch_dims=indices.shape.ndims - 1)


def _batch_gather(params, indices, batch_dims, axis=None):
  r"""Gather slices from params according to indices with leading batch dims.

  This operation assumes that the leading `batch_dims` dimensions of `indices`
  and `params` are batch dimensions; and performs a `tf.gather` operation within
  each batch. (If `batch_dims` is not specified, then it defaults to
  `rank(indices)-1`.)  In the case in which `batch_dims==0`, this operation
  is equivalent to `tf.gather`.

  Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
      tensor. Must be in range `[0, params.shape[batch_dims]]`.
    batch_dims: An integer or none.  The number of batch dimensions.  Must be
      less than `rank(indices)`.  Defaults to `rank(indices) - 1` if None.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`. The
      `axis` in `params` to gather `indices` from. Must be greater than or equal
      to `batch_dims`.  Defaults to the first non-batch dimension. Supports
      negative indexes.

  Returns:
    A Tensor. Has the same type as `params`.

  Raises:
    ValueError: if `indices` has an unknown shape.
  """
  if batch_dims is not None and not isinstance(batch_dims, int):
    raise TypeError("batch_dims must be an int; got %r" % (batch_dims,))
  indices = tf.convert_to_tensor(indices, name="indices")
  params = tf.convert_to_tensor(params, name="params")

  indices_ndims = indices.shape.ndims
  if indices_ndims is None:
    raise ValueError("tf.gather does not allow indices with unknown "
                     "rank when batch_dims is specified.")
  if batch_dims is None:
    batch_dims = indices_ndims - 1
  if batch_dims < 0:
    batch_dims += indices_ndims
  if batch_dims < 0 or batch_dims >= indices_ndims:
    raise ValueError("batch_dims = %d must be less than rank(indices) = %d" %
                     (batch_dims, indices_ndims))
  if params.shape.ndims is not None and batch_dims >= params.shape.ndims:
    raise ValueError("batch_dims = %d must be less than rank(params) = %d" %
                     (batch_dims, params.shape.ndims))

  # Handle axis by transposing the axis dimension to be the first non-batch
  # dimension, recursively calling batch_gather with axis=0, and then
  # transposing the result to put the pre-axis dimensions before the indices
  # dimensions.
  if axis is not None and axis != batch_dims:
    # Adjust axis to be positive.
    if not isinstance(axis, int):
      # axis = tf.where(axis < 0, axis + array_ops.rank(params), axis)
      axis = tf.where(axis < 0, axis + tf.rank(params), axis)
    elif axis < 0 and params.shape.ndims is None:
      # axis = axis + array_ops.rank(params)
      axis = axis + tf.rank(params)
    else:
      if (axis < -params.shape.ndims) or (axis >= params.shape.ndims):
        raise ValueError("axis (%d) out of range [%d, %d)" %
                         (axis, -params.shape.ndims, params.shape.ndims))
      if axis < 0:
        axis += params.shape.ndims
      if axis < batch_dims:
        raise ValueError("batch_dims = %d must be less than or equal to "
                         "axis = %d" % (batch_dims, axis))

    # Move params[axis] up to params[batch_dims].
    perm = [
        list(range(batch_dims)), [axis],
        tf.range(batch_dims, axis, 1),
        tf.range(axis + 1, tf.rank(params), 1)
    ]
    params = tf.transpose(params, tf.concat(perm, axis=0))

    result = _batch_gather(params, indices, batch_dims=batch_dims)

    # Move the result dimensions corresponding to params[batch_dims:axis]
    # to just before the dimensions corresponding to indices[batch_dims:].
    params_start = indices_ndims + axis - batch_dims
    perm = [
        list(range(batch_dims)),
        tf.range(indices_ndims, params_start, 1),
        list(range(batch_dims, indices_ndims)),
        tf.range(params_start, tf.rank(result), 1)
    ]
    return tf.transpose(result, perm=tf.concat(perm, axis=0))

  indices_shape = tf.shape(indices)
  params_shape = tf.shape(params)
  batch_indices = indices
  indices_dtype = indices.dtype.base_dtype
  accum_dim_value = tf.ones((), dtype=indices_dtype)
  # Use correct type for offset index computation
  casted_params_shape = tf.cast(params_shape, indices_dtype)
  for dim in range(batch_dims, 0, -1):
    dim_value = casted_params_shape[dim - 1]
    accum_dim_value *= casted_params_shape[dim]
    start = tf.zeros((), dtype=indices_dtype)
    step = tf.ones((), dtype=indices_dtype)
    dim_indices = tf.range(start, dim_value, step)
    dim_indices *= accum_dim_value
    dim_shape = tf.stack([1] * (dim - 1) + [dim_value] + [1] * (indices_ndims - dim), axis=0)
    batch_indices += tf.reshape(dim_indices, dim_shape)

  flat_indices = tf.reshape(batch_indices, [-1])
  outer_shape = params_shape[batch_dims + 1:]
  flat_inner_shape = tf.reduce_prod(params_shape[:batch_dims + 1], [0], False)
  # flat_inner_shape = gen_math_ops.prod(params_shape[:batch_dims + 1], [0], False)

  flat_params = tf.reshape(params, tf.concat([[flat_inner_shape], outer_shape], axis=0))
  flat_result = tf.gather(flat_params, flat_indices)
  result = tf.reshape(flat_result, tf.concat([indices_shape, outer_shape], axis=0))
  final_shape = indices.get_shape()[:batch_dims].merge_with(params.get_shape()[:batch_dims])
  final_shape = final_shape.concatenate(indices.get_shape().dims[batch_dims:])
  final_shape = final_shape.concatenate(params.get_shape()[batch_dims + 1:])
  result.set_shape(final_shape)
  return result
