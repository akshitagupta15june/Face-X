import argparse
import os
import time
from glob import glob

import numpy as np
from tqdm import tqdm

import utils


def create_bin():
  args = get_args()
  image_paths = glob('{}/data/CelebA_Segment/*.*'.format(args.root_dir))
  train_image_paths, val_image_paths, _ = utils.make_paths(
      image_paths, os.path.join(args.root_dir, 'data', 'params', args.name, 'image'), args.root_dir)
  # with open('data/CelebA_RGBA.bin', 'wb') as img_f, open('data/CelebA_Landmark.bin', 'wb') as lm_f:
  with open('data/CelebA_RGBA.bin', 'wb') as img_f:
    for p in tqdm(train_image_paths):
      try:
        image = utils.load_image(p, 224, True, False)
        img_f.write(image)
        # lm_f.write(landmark)
        img_f.flush()
        # lm_f.flush()
      except Exception as e:
        print(p)
        print(e)


def read_bin():
  # images = open('data/CelebA_RGBA.bin', 'rb')
  # landmarks = open('data/CelebA_Landmark.bin', 'rb')
  # image_array = np.fromstring(images, dtype=np.float32).reshape(
  #     (-1, 224, 224, 4))
  # landmark_array = np.fromstring(landmarks, dtype=np.float32).reshape(
  #     (-1, 51, 2))
  # print(len(images), len(landmarks))
  batch_size = 4
  with open('data/CelebA_RGBA.bin', 'rb') as im_f, open('data/CelebA_Landmark.bin', 'rb') as lm_f:
    for batch in range(128 // batch_size):
      start = time.time()
      im_len = 4 * batch_size * 224 * 224 * 4
      im_start = batch * im_len
      lm_len = 4 * batch_size * 51 * 2
      lm_start = batch * lm_len

      im_f.seek(im_start)
      im_str = im_f.read(im_len)
      lm_f.seek(lm_start)
      lm_str = lm_f.read(lm_len)

      images = np.fromstring(im_str, dtype=np.float32).reshape((-1, 224, 224, 4))
      landmarks = np.fromstring(lm_str, dtype=np.float32).reshape((-1, 51, 2))
      print(time.time() - start)

  print('Done')


def main():
  create_bin()
  # read_bin()


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root_dir', default='/mnt/d/Codes/gcn_face', help='data root directory')
  parser.add_argument('--name', default='bfm09_face', help='dataset name')
  return parser.parse_args()


if __name__ == '__main__':
  main()
