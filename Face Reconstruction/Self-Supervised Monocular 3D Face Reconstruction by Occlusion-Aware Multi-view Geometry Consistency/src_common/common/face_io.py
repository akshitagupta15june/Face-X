from __future__ import division
import os
import numpy as np
from collections import defaultdict

def write_self_lm(path_info_save, lm_all, inter=","):
    if isinstance(path_info_save, str):
        f_info = open(path_info_save, 'w')
    else:
        f_info = path_info_save

    if isinstance(lm_all, list):
        lm_all = np.array(lm_all)

    f_info.write(str(lm_all.shape[0]))
    f_info.write("\n")
    for i in range(lm_all.shape[0]):
        lm = lm_all[i]
        for j in range(len(lm)):
            lm_xyz = lm[j]
            if j != len(lm)-1:
                f_info.write(("%f"+inter) % (lm_xyz))
            else:
                f_info.write("%f" % (lm_xyz))
        f_info.write('\n')

def parse_self_lm(path_info_save):
    with open(path_info_save) as f_info:
        lines = f_info.readlines()
        lines_lm = lines[1:]
        list_lm = []
        for lm2d in lines_lm:
            xyz = lm2d[:-1].split(',')
            xyz = [float(ele) for ele in xyz]
            list_lm.append(xyz)
    return list_lm

def format_file_list(data_root, split, fmt=None, sort=False):
    with open(data_root + '/%s.txt' % split, 'r') as f:
        frames = f.readlines()

    if sort:
        import re
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [atoi(c) for c in re.split(r'(\d+)', text)]

        frames = sorted(frames, key=natural_keys)


    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1][:-1] for x in frames]

    if fmt is None:
        image_file_list = [os.path.join(data_root, subfolders[i], frame_ids[i] + '.jpg') for i in range(len(frames))]
    else:
        image_file_list = [os.path.join(data_root, subfolders[i], frame_ids[i] + fmt) for i in range(len(frames))]
    cam_file_list = [os.path.join(data_root, subfolders[i], frame_ids[i] + '_info.txt') for i in range(len(frames))]

    return image_file_list, cam_file_list, subfolders, frame_ids

# MFS
def write_self_6DoF(path_info_save, dof, inter=","):
    if isinstance(path_info_save, str):
        f_info = open(path_info_save, 'w')
    else:
        f_info = path_info_save

    # intrinsic
    f_info.write('%f,%f,%f,%f,%f,%f\n' %
        (
            dof[0],dof[1],dof[2],dof[3],dof[4],dof[5]
        )
    )

def parse_self_6DoF(path_info_save, inter=","):
    if isinstance(path_info_save, str):
        f_info = open(path_info_save, 'r')
    else:
        f_info = path_info_save

    dof = f_info.readline()[:-1]
    dof = dof.split(inter)
    dof = [float(p) for p in dof]
    return dof


def write_self_intrinsicMtx(path_info_save, intrinsic, inter=","):
    """
    :param path_info_save:
    :param intrinsic: [focalx focaly u v]
    :param inter:
    :return:
    """
    if isinstance(path_info_save, str):
        f_info = open(path_info_save, 'w')
    else:
        f_info = path_info_save

    # intrinsic
    f_info.write('%f,0.,%f,0.,%f,%f,0.,0.,1.\n' %
        (
            intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        )
    )

def parse_self_intrinsicMtx(path_info_save, inter=","):
    if isinstance(path_info_save, str):
        f_info = open(path_info_save, 'w')
    else:
        f_info = path_info_save

    intrin_mtx = f_info.readline()[:-1]
    intrin_mtx = intrin_mtx.split(inter)
    intrin_mtx = [float(p) for p in intrin_mtx]
    intrin_mtx = np.array(intrin_mtx)
    intrin_mtx = np.reshape(intrin_mtx, [3,3])
    return intrin_mtx

#
def write_self_camera(path_info_save, img_width, img_height, intrinsic, pose):
    """
    :param path_info_save: str
    :param intrinsic: shape=[3, 3]
    :param pose: shape=[6], rx, ry, rz, tx, ty, tz
    :return:
    """
    if isinstance(path_info_save, str):
        f_info = open(path_info_save, 'w')
    else:
        f_info = path_info_save

    if len(intrinsic.shape) == 2:
        intrinsic = np.reshape(intrinsic, [-1])

    f_info.write(str(img_width))
    f_info.write(" ")
    f_info.write(str(img_height))
    f_info.write("\n")

    f_info.write("intrinsic")
    f_info.write("\n")
    for i in range(intrinsic.shape[0]):
        row = intrinsic[i]
        if i != len(intrinsic) - 1:
            f_info.write("%f," % (row))
        else:
            f_info.write("%f" % (row))
    f_info.write('\n')

    f_info.write("external")
    f_info.write("\n")
    for i in range(pose.shape[0]):
        row = pose[i]
        if i != len(pose) - 1:
            f_info.write("%f," % (row))
        else:
            f_info.write("%f" % (row))
    f_info.write('\n')

def parser_self_camera(path_info_save):
    """
    :param path_info_save: str
    :param intrinsic: shape=[3, 3]
    :param pose: shape=[6], rx, ry, rz, tx, ty, tz
    :return:
    """
    f_info = open(path_info_save, 'r')

    frs = f_info.readline()
    img_width = int(frs[0])
    img_height = int(frs[1])

    f_info.readline()
    intrin_mtx = parse_self_intrinsicMtx(f_info)

    f_info.readline()
    pose = parse_self_6DoF(f_info)

    return img_width, img_height, intrin_mtx, pose
