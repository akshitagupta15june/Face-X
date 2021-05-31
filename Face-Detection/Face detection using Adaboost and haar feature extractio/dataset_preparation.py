import os
import glob
import sys
import numpy as np 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from math import cos, sin, atan2, degrees, sqrt

path = '/home/varunharis/Desktop/Computer Vision/Project 1/FDDB-folds/'
pathToImages = '/home/varunharis/Desktop/Computer Vision/Project 1/originalPics/'
faceSavePath = '/home/varunharis/Desktop/Computer Vision/Project 1/originalPics/face_train/'
nonFaceSavePath = '/home/varunharis/Desktop/Computer Vision/Project 1/originalPics/non_face_train/'

faceTestPath = '/home/varunharis/Desktop/Computer Vision/Project 1/originalPics/face_test/'
nonFaceTestPath = '/home/varunharis/Desktop/Computer Vision/Project 1/originalPics/non_face_test/'

class prepareDataset(): 
	def __init__(self):
		self.lines = []
		self.Dict = {}
		self.vertices = []

		self.trainImgNumber = 1000
		self.testImgNumber = 100

		self.ftr = np.zeros([self.trainImgNumber,20*20])
		self.ntr = np.zeros([self.trainImgNumber,20*20])

		self.fts = np.zeros([self.testImgNumber,20*20])
		self.nts = np.zeros([self.testImgNumber,20*20])


	def getData(self):
		k = 0
		f=open(path+"FDDB-fold-01-ellipseList.txt","r")

		for line in f:
			self.lines.append(line)      
 
		while k < len(self.lines):
			data=[]
			i=int(self.lines[k+1])
			for j in range(k+2,k+2+i):
				data.append([])
				data[j-(k+2)].append(self.lines[j])

			self.Dict[self.lines[k]] = data
			k = k+i+2

		f.close()

		f = open(path+"FDDB-fold-02-ellipseList.txt","r")

		for line in f:
			self.lines.append(line)      
 
		while k < len(self.lines):
			data=[]
			i=int(self.lines[k+1])
			for j in range(k+2,k+2+i):
				data.append([])
				data[j-(k+2)].append(self.lines[j])

			self.Dict[self.lines[k]] = data
			k = k+i+2

		f.close()

		######## Testing data
		f = open(path+"FDDB-fold-03-ellipseList.txt","r")

		for line in f:
			self.lines.append(line)      
 
		while k < len(self.lines):
			data=[]
			i=int(self.lines[k+1])
			for j in range(k+2,k+2+i):
				data.append([])
				data[j-(k+2)].append(self.lines[j])

			self.Dict[self.lines[k]] = data
			k = k+i+2

		f.close()

	def getFace(self):
		count = 1
		for key, value in self.Dict.items():
			## Value --> 2D list corresponding to the image (one string in each list)
			image = np.array(cv2.imread(pathToImages+key[:-1]+'.jpg'))

			## value[0][0] has the functions in the order as follows
			## rMajor, rMinor, angle, center_x, center_y, 1
			for k in range(len(value)):
				data = value[k][0].split()

				for i in range(len(data)):
					data[i] = float(data[i]) 

				# dataEllipse[2] = data[2]*180/np.pi
				# 4 points of a rectangle
				pt1 = (data[3]+data[0]*cos(data[2])*sqrt(2), 
					   data[4]+data[0]*sin(data[2])*sqrt(2))
				pt2 = (data[3]+data[0]*sin(data[2])*sqrt(2), 
					   data[4]+data[0]*cos(data[2])*sqrt(2))
				pt3 = (data[3]-data[0]*cos(data[2])*sqrt(2), 
					   data[4]-data[0]*sin(data[2])*sqrt(2))
				pt4 = (data[3]-data[0]*sin(data[2])*sqrt(2), 
					   data[4]-data[0]*cos(data[2])*sqrt(2))
				

				## Fit the rectangle bounding the face
				rect = cv2.minAreaRect(np.asarray([pt1,pt2,pt3,pt4], np.float32))
				box = cv2.boxPoints(rect)
				box = np.int0(box)

				imCrop, imRot = self.cropImage(image, rect)
				imCrop = cv2.resize(imCrop, (20,20), interpolation=cv2.INTER_AREA)
				if(count > 1034):
					cv2.imwrite(faceTestPath+'image_'+str(count-1034)+'.jpg', imCrop)
				else:
					cv2.imwrite(faceSavePath+'image_'+str(count)+'.jpg', imCrop)

				# Extract the non face patches
				dim = image.shape
				nonFacePatch = image[int(dim[0]/2)-50:int(dim[0]/2)+51, dim[1]-100:dim[1]+1]
				nonFacePatch = cv2.resize(nonFacePatch, (20,20), interpolation=cv2.INTER_AREA)
				if(count > 1034):
					cv2.imwrite(nonFaceTestPath+'non_face_image_'+str(count-1034)+'.jpg', nonFacePatch)
				else:
					cv2.imwrite(nonFaceSavePath+'non_face_image_'+str(count-1034)+'.jpg', nonFacePatch)
				
				count = count+1

	def cropImage(self, img, rect):
		center = rect[0]
		size = rect[1]
		angle = rect[2]

		center, size = tuple(map(int, center)), tuple(map(int, size))
		rows, cols = img.shape[0], img.shape[1]

		M = cv2.getRotationMatrix2D(center, angle, 1)
		img_rot = cv2.warpAffine(img, M, (cols, rows))
		out = cv2.getRectSubPix(img_rot, size, center)

		return out, img_rot

	def createImgArray(self):
		for i in range(1, self.trainImgNumber):
			face = cv2.imread(faceSavePath+'image_'+str(i)+'.jpg',0)
			nonFace = cv2.imread(nonFaceSavePath+'non_face_image_'+str(i)+'.jpg',0)

			self.ftr[i,:] = face.flatten()
			self.ntr[i,:] = nonFace.flatten()

		for i in range(1, self.testImgNumber):
			testFace = cv2.imread(faceTestPath+'image_'+str(i)+'.jpg',0)
			testNonFace = cv2.imread(nonFaceTestPath+'non_face_image_'+str(i)+'.jpg',0)

			self.fts[i,:] = testFace.flatten()
			self.nts[i,:] = testNonFace.flatten()
