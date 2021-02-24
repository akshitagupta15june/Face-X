from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np

from src_common.common.tf_io import unpack_image_sequence, data_augmentation_mul

class DataLoader(object):
    def __init__(self, 
                 dataset_dir=None,
                 batch_size=None,
                 img_height=None, 
                 img_width=None, 
                 num_source=None,
                 aug_crop_size=None,
                 read_pose=False,
                 match_num=0,
                 read_gpmm=False,
                 flag_data_aug=False,
                 flag_shuffle=True):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.aug_crop_size = aug_crop_size
        self.read_pose = read_pose
        self.match_num = match_num
        self.flag_data_aug = flag_data_aug
        self.flag_shuffle = flag_shuffle

    def format_file_list(self, data_root, split):
        all_list = {}
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        flag_slg_mul = [x.split(' ')[0] for x in frames]
        subfolders = [x.split(' ')[1] for x in frames]
        frame_ids = [x.split(' ')[2][:-1] for x in frames]

        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        skin_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_skin.jpg') for i in range(len(frames))]

        steps_per_epoch = int(len(image_file_list) // self.batch_size)
        print("*************************************************************** format_file_list ")
        img_cam = list(zip(image_file_list, cam_file_list, skin_file_list, flag_slg_mul))
        if self.flag_shuffle:
            random.shuffle(img_cam)
        else:
            print("Without shuffle")
        image_file_list, cam_file_list, skin_file_list, flag_slg_mul = zip(*img_cam)

        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['skin_file_list'] = skin_file_list
        all_list['flag_slg_mul'] = flag_slg_mul

        self.steps_per_epoch = int(len(all_list['image_file_list']) // self.batch_size)
        print("Finish format_file_list")
        if len(image_file_list) > 10:
            for i in range(10):
                print(image_file_list[i])
        return all_list

    def load_train_batch(self, is_test=False):
        """
        Load a batch of training instances using the new tensorflow
        Dataset api.
        """
        def _parse_train_img(img_path):
            with tf.device('/cpu:0'):
                img_buffer = tf.read_file(img_path)
                image_decoded = tf.image.decode_jpeg(img_buffer)
                # TODO: TW Image sequence structure
                tgt_image, src_image_stack = \
                     unpack_image_sequence(
                        image_decoded, self.img_height, self.img_width, self.num_source)
            return tgt_image, src_image_stack

        def _batch_preprocessing(stack_images, stack_images_skin, flag_sgl_mul, intrinsics, optional_data1, optional_data2):
            intrinsics = tf.cast(intrinsics, tf.float32)
            image_all = tf.concat([stack_images[0], stack_images[1], stack_images_skin[0], stack_images_skin[1]], axis=3)

            if not is_test and self.flag_data_aug:  # otherwise matches coords are wrong
                if self.match_num == 0:
                    image_all, intrinsics, matches = data_augmentation_mul(
                        image_all, intrinsics, self.img_height, self.img_width)
                else:
                    image_all, intrinsics, matches = data_augmentation_mul(
                        image_all, intrinsics, self.img_height, self.img_width, optional_data2)
            else:
                matches = optional_data2

            image = image_all[:, :, :, :3*(self.num_source+1)]
            image_skin = image_all[:, :, :, 3*(self.num_source+1):]
            return image, image_skin, flag_sgl_mul, intrinsics, optional_data1, matches

        input_image_names_ph = tf.placeholder(tf.string, shape=[None], name='input_image_names_ph')
        image_dataset = tf.data.Dataset.from_tensor_slices(input_image_names_ph).map(_parse_train_img)

        input_skin_names_ph = tf.placeholder(tf.string, shape=[None], name='input_skin_names_ph')
        skin_dataset = tf.data.Dataset.from_tensor_slices(input_skin_names_ph).map(_parse_train_img)

        flag_sgl_mul_ph = tf.placeholder(tf.int32, [None], name='flag_sgl_mul_ph')
        flag_sgl_mul_dataset = tf.data.Dataset.from_tensor_slices(flag_sgl_mul_ph)

        cam_intrinsics_ph = tf.placeholder(tf.float32, [None, 1+self.num_source, 3, 3], name='cam_intrinsics_ph')
        intrinsics_dataset = tf.data.Dataset.from_tensor_slices(cam_intrinsics_ph)

        datasets = (image_dataset, skin_dataset, flag_sgl_mul_dataset, intrinsics_dataset)
        if self.read_pose:
            poses_ph = tf.placeholder(tf.float32, [None, 1+self.num_source, 6], name='poses_ph')
            pose_dataset = tf.data.Dataset.from_tensor_slices(poses_ph)
            datasets = datasets + (pose_dataset,)
        else:
            datasets = datasets + (intrinsics_dataset,)
        if self.match_num > 0:
            matches_ph = tf.placeholder(tf.float32, [None, (1+self.num_source), self.match_num, 2], name='matches_ph')
            match_dataset = tf.data.Dataset.from_tensor_slices(matches_ph)
            datasets = datasets + (match_dataset,)
        else:
            datasets = datasets + (intrinsics_dataset,)

        all_dataset = tf.data.Dataset.zip(datasets)
        all_dataset = all_dataset.batch(self.batch_size)
        if self.flag_shuffle:
            all_dataset = all_dataset.shuffle(buffer_size=4000).repeat().prefetch(self.batch_size*4)
        all_dataset = all_dataset.map(_batch_preprocessing)
        iterator = all_dataset.make_initializable_iterator()
        return iterator

    def init_data_pipeline(self, sess, batch_sample, file_list):
        #
        def _load_cam_intrinsics(cam_filelist, read_pose, match_num):
            all_cam_intrinsics = []
            all_poses = []
            all_matches = []

            for i in range(len(cam_filelist)):
                filename = cam_filelist[i]
                if i % 50000 == 0:
                    print(i, ' in all: ', len(cam_filelist))
                f = open(filename)
                one_intrinsic = []
                for i in range(1 + self.num_source):
                    line = f.readline()
                    #
                    cam_intri_vec = [float(num) for num in line.strip().split(',')]
                    if len(cam_intri_vec) != 9:
                        print(filename, i, line)

                    cam_intrinsics = np.reshape(cam_intri_vec, [3, 3])
                    one_intrinsic.append(cam_intrinsics)
                one_intrinsic = np.stack(one_intrinsic, axis=0)
                all_cam_intrinsics.append(one_intrinsic)
                #
                if read_pose:
                    one_sample_pose = []
                    for i in range(0, 1 + self.num_source):
                        lines = f.readline()
                        pose = [float(num) for num in lines.strip().split(',')]
                        pose_vec = np.reshape(pose, [6])
                        one_sample_pose.append(pose_vec)
                    one_sample_pose = np.stack(one_sample_pose, axis=0)
                    all_poses.append(one_sample_pose)
                #
                if match_num > 0:
                    image_matches = []
                    for i in range(1 + self.num_source):
                        one_matches = []
                        line = f.readline()
                        for j in range(self.match_num):
                            line = f.readline()

                            match_coords = [float(num) for num in line.strip().split(',')]
                            match_vec = np.reshape(match_coords, [2])
                            one_matches.append(match_vec)
                        one_matches = np.stack(one_matches, axis=0) # 68
                        image_matches.append(one_matches)
                        # TODO: Very dangerous
                        # if i == self.num_source / 2:
                        #     image_matches = [one_matches] + image_matches
                        # else:

                    image_matches = np.stack(image_matches, axis=0) # (1 + self.num_source), 68

                    all_matches.append(image_matches)

                f.close()
            all_cam_intrinsics = np.stack(all_cam_intrinsics, axis=0)

            if read_pose:
                all_poses = np.stack(all_poses, axis=0)
            if match_num > 0:
                all_matches = np.stack(all_matches, axis=0)
            return all_cam_intrinsics, all_poses, all_matches

        # load cam_intrinsics using native python
        print('load camera intrinsics...')
        cam_intrinsics, all_poses, all_matches = _load_cam_intrinsics(file_list['cam_file_list'], self.read_pose, self.match_num)

        input_dict = {'data_loading/input_image_names_ph:0': file_list['image_file_list'][:self.batch_size * self.steps_per_epoch],
                      'data_loading/input_skin_names_ph:0': file_list['skin_file_list'][:self.batch_size * self.steps_per_epoch],
                      'data_loading/flag_sgl_mul_ph:0': file_list['flag_slg_mul'][:self.batch_size * self.steps_per_epoch],
                      'data_loading/cam_intrinsics_ph:0': cam_intrinsics[:self.batch_size*self.steps_per_epoch]}
        if self.read_pose:
            print('load pose data...')
            input_dict['data_loading/poses_ph:0'] = all_poses[:self.batch_size*self.steps_per_epoch]
        if self.match_num > 0:
            print('load matches data...')
            input_dict['data_loading/matches_ph:0'] = all_matches[:self.batch_size*self.steps_per_epoch]

        sess.run(batch_sample.initializer, feed_dict=input_dict)

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, 0, tgt_start_idx, 0], 
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0, 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, 0, int(tgt_start_idx + img_width), 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, 0, i*img_width, 0], 
                                    [-1, -1, img_width, -1]) 
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

if __name__ == "__main__":
    path_data = "/home/jshang/SHANG_Data_MOUNT/141/GAFR_semi_bfmAlign_3/11141_300WLP_CelebA_Mpie_tensor_MERGE"
    h_loader = DataLoader(batch_size=5)
    f = h_loader.format_file_list(path_data, 'train', 3, 2)