# encoding:utf-8

# This method is for calculating serveral usual errors of landmark detection

num = 5
import numpy as np
file = open('testset/test.txt','r')
img_list = file.readlines()
file.readlines()

errors_norm = []
errors_pupil = []
errors_ocular = []
errors_diagnal = []
for i in range(num):
    img_name = img_list[i].strip('\n')
    land2d = np.load('testset/landmarks/'+img_name+'_l.npy')
    gt_land = np.load('testset/landmarks_gt/'+img_name+'.jpg.npy')
    error = np.mean(np.sqrt(np.sum((land2d-gt_land)**2, axis=1)))
    pupil_norm = np.linalg.norm(np.mean(gt_land[36:42], axis=0) - np.mean(gt_land[42:48], axis=0))
    ocular_norm = np.linalg.norm(gt_land[36] - gt_land[45])
    height, width = np.max(gt_land, axis=0) - np.min(gt_land, axis=0)
    diagnal_norm = np.sqrt(height**2 + width**2)
    errors_norm.append(error)
    errors_pupil.append(error/pupil_norm)
    errors_ocular.append(error/ocular_norm)
    errors_diagnal.append(error/diagnal_norm)
print("the mean error: "+str(np.mean(errors_norm)))
print("the mean error(pupil): "+str(np.mean(errors_pupil)))
print("the mean error(ocular): "+str(np.mean(errors_ocular)))
print("the mean error(diagnal): "+str(np.mean(errors_diagnal)))
