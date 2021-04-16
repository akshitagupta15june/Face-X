# coding: utf-8

import cv2
import numpy as np
import glob
import math
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

def dim_reduction(feature_vector_train,feature_vector_test,components):
    '''TRAINING'''
    ft_train=feature_vector_train
    
    #get the classes of all training feature vectors
    y_train=[]
    for i in range(0,108):
        for k in range(0,3):
            y_train.append(i+1)
    y_train=np.array(y_train)
    
    #fit the LDA model on training data with n components
    sklearn_lda = LDA(n_components=components)
    sklearn_lda.fit(ft_train,y_train)
    
    #transform the traning data
    red_train=sklearn_lda.transform(ft_train)
    
    '''TESTING'''
    ft_test=feature_vector_test
    
    #transform the testing data
    red_test=sklearn_lda.transform(ft_test)
    
    
    #get a list of predicted values for the testing data to calculate ROC
    y_pred=sklearn_lda.predict(ft_test)
    
    #return transformed training and testing data, and the testing classes and predicted values for ROC
    return red_train,red_test


def IrisMatching(feature_vector_train,feature_vector_test,components,flag):
    
    #if flag is 1, we do not need to reduce dimesionality otherwise we call the dim_reduction function
    if flag==1:
        red_train=feature_vector_train
        red_test=feature_vector_test
        
    elif flag==0:
        red_train,red_test=dim_reduction(feature_vector_train,feature_vector_test,components)


    arr_f=red_test #test
    arr_fi=red_train #train
    
    index_L1=[]
    index_L2=[]
    index_cosine=[]
    min_cosine=[]
    
    #this loop iterates over each test image
    for i in range(0,len(arr_f)):
        L1=[]
        L2=[]
        Cosine=[]
        
        #this loop iterates over every training image - to be compared to each test image
        for j in range(0,len(arr_fi)):
            f=arr_f[i]
            fi=arr_fi[j]
            sumL1=0 #L1 distance
            sumL2=0 #L2 distance
            sumcos1=0
            sumcos2=0
            cosinedist=0 #cosine distance
            
            #calculate L1 and L2 using the formulas in the paper
            for l in range(0,len(f)):
                sumL1+=abs(f[l]-fi[l])
                sumL2+=math.pow((f[l]-fi[l]),2)
            
            
            #calculate sum of squares of all features for cosine distance
            for k in range(0,len(f)):
                sumcos1+=math.pow(f[k],2)
                sumcos2+=math.pow(fi[k],2)
                
            
            #calculate cosine distance using sumcos1 and sumcos2 calculated above
            cosinedist=1-((np.matmul(np.transpose(f),fi))/(math.pow(sumcos1,0.5)*math.pow(sumcos2,0.5)))
            
            L1.append(sumL1)
            L2.append(sumL2)
            Cosine.append(cosinedist)
        #get minimum values for L1 L2 and cosine distance for each test image and store their index
        index_L1.append(L1.index(min(L1)))
        index_L2.append(L2.index(min(L2)))
        index_cosine.append(Cosine.index(min(Cosine)))
        min_cosine.append(min(Cosine))
        
    match=0
    count=0
    
    #stores final matching - correct(1) or incorrect (0)
    match_L1=[]
    match_L2=[]
    match_cosine=[]
    match_cosine_ROC=[]
    
    #calculating matching of the test set according to the ROC thresholds
    thresh=[0.4,0.5,0.6]
    
    for x in range(0,len(thresh)):
        match_ROC=[]
        for y in range(0,len(min_cosine)):
            if min_cosine[y]<=thresh[x]:
                match_ROC.append(1)
            else:
                match_ROC.append(0)
        match_cosine_ROC.append(match_ROC)
        
    
    for k in range(0,len(index_L1)):
        '''count goes from 0 to 3 because we compare the indexes obtained for the first 4 images of the test data
        to the indexes of the first 3 images of 
        the train data (for which match is incremented by 3 everytime count exceeds the value of 3)'''
        if count<4:
            count+=1
        else:
            match+=3
            count=1
            
        '''check if matching is done correctly (1) or not (0) for L1 L2 and cosine distance and accordingly update
        the arrays match_L1,match_L2,match_cosine'''
        if index_L1[k] in range(match,match+3):
                match_L1.append(1)
        else:
            match_L1.append(0)
        if index_L2[k] in range(match,match+3):
            match_L2.append(1)
        else:
            match_L2.append(0)
        if index_cosine[k] in range(match,match+3):
            match_cosine.append(1)
        else:
            match_cosine.append(0)
    #reuturns the matching arrays, and test calsses and predicted values to calculate ROC
    return match_L1,match_L2,match_cosine,match_cosine_ROC

