# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:58:28 2020

@author: varun
"""

import cv2
import numpy as np

from dask import delayed
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from sklearn.model_selection import train_test_split

facePath = "E:\\NCSU\\Spring-20\\Computer Vision\\Project\\Project 2\\data\\Face16\\"
nonFacePath = "E:\\NCSU\\Spring-20\\Computer Vision\\Project\\Project 2\\data\\Nonface16\\"

class haarFeatures(object):
    def __init__(self):
        self.trainImages = []
        self.labels = np.array([1] * 400 + [-1] * 800)
        self.createImgArray()
    
    def createImgArray(self):
        for i in range(400):
            self.trainImages.append(cv2.imread(facePath+'c000'+str(i+100)+'.bmp',0))
        
        for i in range(800):
            self.trainImages.append(cv2.imread(nonFacePath+str(i)+'.bmp',0))
            
        self.trainImages = np.array(self.trainImages)
                    
    @delayed
    def extract_feature_image(self, img, feature_type, feature_coord=None):
        # Extract the haar feature for the current image
        # Integral image is computed for optimization of convolution operation
        ii = integral_image(img)
        return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1], feature_type=feature_type, feature_coord=feature_coord)
    
    def _get_features(self):
        print('Extracting Haar Features, it may take a while !!')
        feature_types = ['type-2-x', 'type-2-y', 'type-3-x','type-3-y','type-4']
        # Build a computation graph using Dask. This allows the use of multiple
        # CPU cores later during the actual computation
        X = delayed(self.extract_feature_image(img, feature_types) for img in self.trainImages)
        X = np.array(X.compute(scheduler='threads'), dtype=np.uint8)
        
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(X,self.labels,train_size=1000,
                                                                           random_state=0,
                                                                           stratify=self.labels)