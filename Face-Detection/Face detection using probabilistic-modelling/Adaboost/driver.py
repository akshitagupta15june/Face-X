# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 02:56:50 2020

@author: varun
"""
import numpy as np
import matplotlib.pyplot as plt
from haar_features import haarFeatures
from ada_boost import adaBoost
import tensorflow as tf
from skimage.feature import haar_like_feature_coord, draw_haar_like_feature

def getScore(label, prediction):
    return np.sum(label == prediction, axis=0)/len(label)

if __name__ == '__main__':
    # GPU usage for optimization purposes    
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    
    haar = haarFeatures()
    boost = adaBoost(10)
    
    haar._get_features()
    boost.train(haar.trainX, haar.trainY)
    print('Training Done !!')

    # Coordinates of every haar feature in the training set
    feature_coord, feature_type = haar_like_feature_coord(width = 16, height = 16,
                                                          feature_type=['type-2-x', 'type-2-y', 'type-3-x','type-3-y','type-4'])
    
    # Running on test set
    ans = input('Want to test? [y]/n: ')    
    if(ans == 'y'):
        tpr = []
        fpr = []
        
        # Threshold range computation for ROC
        alphaSum=0
        for clf in boost.clfs:
            alphaSum += clf[3]    
        lb =- alphaSum*2
        ub =+ alphaSum*2
        step = 0.1
        threshold = np.arange(lb,ub,step)
        
        for t in range(threshold.size):
            yPred = boost.predict(haar.testX, threshold[t])
            tpr.append(np.sum((yPred == 1)*(haar.testY == 1))/np.sum(haar.testY == 1))
            fpr.append(np.sum((haar.testY == -1)*(yPred == 1))/np.sum(haar.testY == -1))            
        
        # Plotting best features before boosting
        fig1, axes1 = plt.subplots(5, 2)
        for idx, ax in enumerate(axes1.ravel()):
            image = draw_haar_like_feature(haar.trainImages[0], 0, 0,
                                           haar.trainImages.shape[2],
                                           haar.trainImages.shape[1],
                                           [feature_coord[boost.beforeBoost[idx][1]]])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
                
        # Plotting features after boosting
        fig2, axes2 = plt.subplots(5, 2)
        for idx, ax in enumerate(axes2.ravel()):
            image = draw_haar_like_feature(haar.trainImages[0], 0, 0,
                                           haar.trainImages.shape[2],
                                           haar.trainImages.shape[1],
                                           [feature_coord[boost.clfs[idx][1]]])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.figure()            
        plt.plot(np.array(fpr), np.array(tpr))
        plt.show()
            
        yPred = boost.predict(haar.testX, 0)
        accuracy = getScore(haar.testY, yPred)
        print('Accuracy = '+str(accuracy))