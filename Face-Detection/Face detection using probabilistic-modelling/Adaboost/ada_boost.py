# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 01:50:32 2020

@author: varun
"""
import numpy as np
import math
    
class adaBoost(object):
    def __init__(self, weakClf=5):
        self.weakClf = weakClf
        self.thresh = None
        self.alpha = None
        self.polarity = 1
        self.feature = None
        self.beforeBoost = []
        
    def train(self, X, labels):
        nImgs, nFeatures = X.shape
        # Weight intialization
        weights = np.full(nImgs, (1/nImgs), dtype=float)
        # List that jolds all the weak classifiers (Each has a list of parameters)
        self.clfs = list()
        
        # Iterates through each weak classifier to get its parameters
        for i in range(self.weakClf):
            print('Evaluating weak classifier number '+str(i))
            min_error = float('inf')
            
            # Every feature is a weak learner, a weak classifier is identified by its best weak learner
            for ftr in range(nFeatures):
                values = np.expand_dims(X[:,ftr], axis=1)
                # We take the unique values for a feature to remove any redundant computations
                for threshold in np.unique(values):
                    p = 1
                    pred = np.ones(labels.shape)
                    pred[X[:,ftr] < threshold] = -1
                    
                    error = np.sum(weights[labels != pred])
                    if error > 0.5:
                        error = 1 - error 
                        p = -1
                        
                    if error < min_error:
                        self.polarity = p
                        self.thresh = threshold
                        self.feature = ftr
                        min_error = error
                        
                        if i==0:
                            self.beforeBoost.append((min_error, ftr))
                        
            self.alpha = 0.5*math.log((1. - min_error)/(min_error + 1e-10))
            predictions = np.ones(labels.shape)
            neg_idx = (self.polarity * X[:,self.feature] < \
                       self.polarity * self.thresh)
            
            if i == 0:
                self.getFeaturesBeforeBoost()
            
            predictions[neg_idx] = -1            
            weights *= np.exp((-1)*self.alpha*labels*predictions)
            weights /= np.sum(weights)
            
            self.clfs.append([self.polarity, self.feature, self.thresh, self.alpha])
            # Once the weak classifier is identified, we can reset the parameters for the next iteration
            self.feature = None
            self.thresh = None
            self.alpha = None
            self.polarity = 1
            
            print('Weak classifier number '+str(i)+' done')
            
    def getFeaturesBeforeBoost(self):
        print('Fetching top 10 features before boosting')
        self.beforeBoost.sort()
        temp = self.beforeBoost
        self.beforeBoost = temp[0:10]
        print('Back to Adaboost Algorithm')
            
    def predict(self, X, threshold):
        nImgs = X.shape[0]
        yPred = np.zeros((nImgs,1))
        
        # Strong classifier is an  amalgamation of all the weak classifiers
        # It takes into account polarity, weights and feature along with the prediction for each classifier
        for clf in self.clfs:
          pred = np.ones(yPred.shape)
          neg_idx = (clf[0] * X[:,clf[1]] < \
                     clf[0] * clf[2])
          
          pred[neg_idx] = -1
          yPred += clf[3]*pred
        
        return np.sign((yPred-threshold).flatten())