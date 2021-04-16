
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

def IrisLocalization(images):
    #convert image to a color image
    target = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    boundary=[] #initialize empty list that will eventually contain all the images with boundaries
    centers=[] #initialize empty list that will contain the centers of the boundary circles
    for img in target:
        
        draw_img=img
        
        # remove noise by blurring the image
        blur = cv2.bilateralFilter(img, 9,75,75)
        img=blur
        
        #estimate the center of pupil
        horizontalProjection = np.mean(img,0);
        verticalProjection = np.mean(img,1);
        center_x=horizontalProjection.argmin()
        center_y=verticalProjection.argmin()
        
        #recalculate of pupil by concentrating on a 120X120 area
        centrecrop_x = img[center_x-60:center_x+60]
        centrecrop_y = img[center_y-60:center_y+60]
        horizontalProjection = np.mean(centrecrop_y,0);
        verticalProjection = np.mean(centrecrop_x,0);
        crop_center_x=horizontalProjection.argmin()
        crop_center_y=verticalProjection.argmin()

        cimg=img.copy()
        cv2.circle(cimg,(crop_center_x,crop_center_y),1,(255,0,0),2)

        #apply Canny edge detector on the masked image
        maskimage = cv2.inRange(img, 0, 70)
        output = cv2.bitwise_and(img, maskimage)
        edged = cv2.Canny(output, 100, 220)
        
        # Apply Hough transform to find potential boundaries of pupil
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 10, 100)
        
        #define the center of the pupil
        a = (crop_center_x,crop_center_y)
        
        out = img.copy()
        min_dst=math.inf
        for i in circles[0]:
            #find the circle whose center is closest to the approx center found above
            b=(i[0],i[1])
            dst = distance.euclidean(a, b)
            if dst<min_dst:
                min_dst=dst
                k=i
                
        #draw the inner boundary
        cv2.circle(draw_img, (k[0], k[1]), k[2], (255, 0, 0), 3)

        pupil=circles[0][0]
        radius_pupil = int(k[2])
        
        #draw the outer boundary, which is approximately found to be at a distance 53 from the inner boundary 
        cv2.circle(draw_img, (k[0], k[1]), radius_pupil+53, (255, 0, 0), 3)
        boundary.append(draw_img)
        centers.append([k[0],k[1],k[2]])
    return boundary,centers

