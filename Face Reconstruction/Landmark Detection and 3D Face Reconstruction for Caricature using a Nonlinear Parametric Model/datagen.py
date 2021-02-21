"""
This file is for data preprocessing
"""
import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

train_transform=transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

def clip_rot_flip_face(image, landmarks, scale, alpha):
    """
        given a face and its 68 landmarks,
        clip and rotate it with 'scale' and 'alpha',
        and return scaled and rotated face with new 68 landmarks.
    """
    # calculate x_min, x_max, y_min, y_max, center_x, center_y and box_w
    x_min = np.min(landmarks[:,0])
    x_max = np.max(landmarks[:,0])
    y_min = np.min(landmarks[:,1])
    y_max = np.max(landmarks[:,1])
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    box_w = max(x_max-x_min, y_max-y_min)

    # calculate x_0, y_0, x_1 and y_1
    x_0 = np.random.uniform(center_x-(0.5+0.75*(scale-1.0))*box_w, center_x-(0.5+0.25*(scale-1.0))*box_w)
    y_0 = np.random.uniform(center_y-(0.5+0.75*(scale-1.0))*box_w, center_y-(0.5+0.25*(scale-1.0))*box_w)
    x_0 = round(max(0.0, x_0))
    y_0 = round(max(0.0, y_0))
    x_1 = round(min(image.shape[1], x_0+scale*box_w))
    y_1 = round(min(image.shape[0], y_0+scale*box_w))
    
    """
        process image and landmarks
    """
    # Random clip for image
    new_image = image[int(y_0):int(y_1), int(x_0):int(x_1)]
    new_image = cv2.resize(new_image, (224,224), interpolation=cv2.INTER_LINEAR)
    # Random rotate for image
    rot_mat = cv2.getRotationMatrix2D((112, 112), alpha, 1) # obtain RotationMatrix2D with fixed center (112, 112)
    new_image = cv2.warpAffine(new_image, rot_mat, (224, 224)) # obtain rotated image
    
    # Random clip for corresponding landmarks
    new_landmarks = landmarks
    new_landmarks[:,0] = (new_landmarks[:,0] - x_0) * 224 / (x_1 - x_0)
    new_landmarks[:,1] = (new_landmarks[:,1] - y_0) * 224 / (y_1 - y_0)
    # Random rotate for corresponding landmarks
    new_landmarks = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
    rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in new_landmarks]) # adjust new_landmarks after rotating
    
    return new_image, new_landmarks

class TrainSet(data.Dataset):
    """
        construct trainset, including images, landmarks, and vertices of ground-truth meshes
    """
    def __init__(self, image_path, landmark_path, vertex_path, landmark_num=68, vertex_num=6144):
        """
            initialize TrainSet
        """
        file = open(image_path,'r')
        image = file.readlines()
        file.close()
        file = open(landmark_path,'r')
        landmark = file.readlines()
        file.close()
        file = open(vertex_path,'r')
        vertex = file.readlines()
        file.close()
        self.image = [os.path.join(k.strip('\n')) for k in image]
        self.landmark = [os.path.join(k.strip('\n')) for k in landmark]
        self.vertex = [os.path.join(k.strip('\n')) for k in vertex]
        self.transforms = train_transform
        if len(self.image) == len(self.landmark) == len(self.vertex):
            self.num_samples = len(self.image)
        self.landmark_arrays = np.zeros((self.num_samples, landmark_num, 2), np.float32)
        self.vertex_arrays = np.zeros((self.num_samples, 3, vertex_num), np.float32)
        for i in range(self.num_samples):
            self.landmark_arrays[i,...] = np.load(self.landmark[i])
            self.vertex_arrays[i,...] = np.load(self.vertex[i])

    def __getitem__(self,index):
        # get image
        image_path = self.image[index]
        image = cv2.imread(image_path)
        # get landmark
        landmark = self.landmark_arrays[index,...]
        # get vertex
        vertex = self.vertex_arrays[index,...]

        """
            preprocess image and landmark
        """
        # image, landmark = clip_rot_flip_face(image, landmark, 1.2, np.random.uniform(-10.0, 10.0))
        image, landmark = clip_rot_flip_face(image, landmark, 1.2, 5*np.random.randint(-1, 2))
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image)

        return image, landmark, vertex

    def __len__(self):
        return self.num_samples

class TestSet(data.Dataset):
    """
        construct test set,
        including images, landmarks for calculating errors
        and 'lrecord', 'vrecord' for recording estimated landmarks and vertices of recovered meshes.
    """
    def __init__(self, image_path, landmark_path, lrecord_path, vrecord_path):
        """
            initialize Test Set
        """
        file = open(image_path,'r')
        image = file.readlines()
        file.close()
        file = open(landmark_path, 'r')
        landmark = file.readlines()
        file.close()
        file = open(lrecord_path,'r')
        lrecord = file.readlines()
        file.close()
        file = open(vrecord_path,'r')
        vrecord = file.readlines()
        file.close()
        self.image = [os.path.join(k.strip('\n')) for k in image]
        self.landmark = [os.path.join(k.strip('\n')) for k in landmark]
        self.lrecord = [os.path.join(k.strip('\n')) for k in lrecord]
        self.vrecord = [os.path.join(k.strip('\n')) for k in vrecord]
        self.transforms = test_transform
        if len(self.image) == len(self.landmark) == len(self.lrecord) == len(self.vrecord):
            self.num_samples = len(self.image)

    def __getitem__(self,index):
        # get image
        image_path = self.image[index]
        image = Image.open(image_path)
        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image)
        # get landmark
        landmark_path = self.landmark[index]
        landmark = np.load(landmark_path)
        # get record
        lrecord = self.lrecord[index]
        vrecord = self.vrecord[index]

        return image, landmark, lrecord, vrecord

    def __len__(self):
        return self.num_samples