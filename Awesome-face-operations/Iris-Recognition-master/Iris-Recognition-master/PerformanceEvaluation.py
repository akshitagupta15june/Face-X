
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

def PerformanceEvaluation(match_L1,match_L2,match_cosine):
    
    #storing only those elements that are correctly matched
    correct_L1 = [l for l in match_L1 if l==1]
    correct_L2 = [l for l in match_L2 if l==1]
    correct_cosine = [l for l in match_cosine if l==1]
    
    
    #calculating the correct recognition rates for L1,L2 and cosine similarity
    crr_L1=len(correct_L1)/len(match_L1)
    crr_L2=len(correct_L2)/len(match_L2)
    crr_cosine=len(correct_cosine)/len(match_cosine)
    
    
    return crr_L1*100,crr_L2*100,crr_cosine*100 #return CRR percentages