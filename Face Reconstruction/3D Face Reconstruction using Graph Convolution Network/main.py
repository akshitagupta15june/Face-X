import argparse
import os
import time
from glob import glob

import numpy as np
import tensorflow as tf

import utils
from lib.mesh_io import read_obj
from model_normal import Model as NormalModel
from model_resnet import Model as ResnetModel


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', default='bfm09_face', help='dataset name')
  parser.add_argument('--suffix', default=None, help='suffix for training name')
  parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'],
                      help='train or test')
  parser.add_argument('--stage', default='all', choices=['all', 'rec', 'render'],
                      help='training stage, only rec_loss, only render_loss or all_loss')
  parser.add_argument('--restore', default=False, action='store_true',
                      help='restore checkpoint for training')
  parser.add_argument('--gan', default=False, action='store_true', help='using gan or not')
  parser.add_argument('--wide', default=False, action='store_true', help='using gan or not')
  parser.add_argument('--model', default='normal',
                      help='using model, chose from normal, resnet, adv')
  parser.add_argument('--root_dir', default='/data/gcn_face', help='data root directory')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='input batch size for training (default: 64)')
  parser.add_argument('--epoch', type=int, default=50,
                      help='number of epochs to train (default: 2)')
  parser.add_argument('--eval', type=float, default=0, help='eval frequency')
  parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
  parser.add_argument('--nz', type=int, default=512, help='Size of latent variable')
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
  parser.add_argument('--buffer_size', type=int, default=10,
                      help='buffer size for training data loading')
  parser.add_argument('--workers', type=int, default=4, help='number of data loading threads')
  parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
  parser.add_argument('--adv_lambda', type=float, default=1e-3, help='lambda for adversarial loss')
  parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
  parser.add_argument('--input', default='data/test/raw', type=str,
                      help='test input data path or directory')
  parser.add_argument('--output', default='results/raw', type=str,
                      help='test output path or directory')

  return parser.parse_args()


def main():
  args = get_args()
  logger = utils.init_logger()
  logger.info(args)

  np.random.seed(args.seed)
  if not os.path.isdir(args.root_dir):
    # args.root_dir = '.'
    args.root_dir = '/mnt/d/Codes/gcn_face'
  logger.info("Loading data from %s", args.root_dir)

  if args.suffix is None:
    args.suffix = args.model
    if args.gan:
      args.suffix = args.suffix + '_gan'

  refer_mesh = read_obj(os.path.join(args.root_dir, 'data', 'bfm09_face_template.obj'))
  # refer_meshes = utils.get_mesh_list(args.name)

  image_paths = glob('{}/data/CelebA_Segment/*.*'.format(args.root_dir))
  _, val_image_paths, test_image_paths = utils.make_paths(
      image_paths, os.path.join(args.root_dir, 'data', 'params', args.name, 'image'), args.root_dir)

  if args.mode == 'train':
    img_file = open(os.path.join(args.root_dir, 'data', 'CelebA_RGBA.bin'), 'rb')
    # lm_file = open(os.path.join(args.root_dir, 'data', 'CelebA_Landmark.bin'), 'rb')
  else:
    img_file = None

  gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  # pylint: disable=no-member
  gpu_config.gpu_options.allow_growth = True
  with tf.Graph().as_default() as graph, tf.device('/gpu:0'), tf.Session(config=gpu_config) as sess:
    if args.model == 'normal':
      model = NormalModel(args, sess, graph, refer_mesh, image_paths, img_file)
    elif args.model == 'resnet':
      model = ResnetModel(args, sess, graph, refer_mesh, image_paths, img_file)

    if args.mode in ['train']:
      # if not os.path.exists(os.path.join('checkpoints', args.name)):
      #   os.makedirs(os.path.join('checkpoints', args.name))
      model.fit()
      img_file.close()
      # lm_file.close()
    else:
      if args.input and not os.path.isdir(args.input):
        args.input = None
      if args.input is not None:
        # input_dir = os.path.join('data', 'test', args.input)
        input_dir = args.input
        test_image_paths = [os.path.join(input_dir, x) for x in sorted(os.listdir(input_dir))]
        if args.output is None:
          test_dir = os.path.join('results', args.input)
      else:
        if args.output is None:
          test_dir = model.samp_dir + '_test'
      if args.output is not None:
        test_dir = args.output
      if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

      predictor_path = os.path.join('data', 'shape_predictor_68_face_landmarks.dat')
      cropper = utils.ImageCropper(predictor_path, model.img_size)

      test_image = utils.load_images(test_image_paths, model.img_size, False, False, cropper)

      from face_segment import Segment
      segmenter = Segment()
      alphas = segmenter.segment(test_image)

      test_rgba = np.concatenate([test_image, alphas[..., np.newaxis]], axis=-1)

      string, results = model.evaluate(test_rgba)
      logger.info(string)

      for i, path in enumerate(test_image_paths):
        model.save_sample(results, i, test_rgba, None, test_dir, i, False)
        logger.info('Saving results from %s', path)


if __name__ == '__main__':
  main()
