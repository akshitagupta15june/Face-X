import os
import sys
import glob
import numpy as np
from numpy.linalg import inv,det
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset import prepareDataset
import matplotlib.pyplot as plt

pca_dim = 23

class simpleGaussian():
	def __init__(self):
		self.obj1 = prepareDataset()

	def pdf(self, X, i, mean, covariance):
		x = X[:,i] - mean
		term = np.matmul(x.T, inv(covariance))
		num = -0.5*np.matmul(term, x)
		den = np.sqrt(np.power(2*np.pi, len(mean))*det(covariance))
		return 1.*np.exp(num)/den

	def gaussModel(self, t_train_pca, u_train_pca):
		muFace = t_train_pca.mean(axis=1)
		muNonFace = u_train_pca.mean(axis=1)

		varFace = np.diag(np.diag(np.cov(t_train_pca, rowvar=True)))
		varNonFace = np.diag(np.diag(np.cov(u_train_pca, rowvar=True)))

		return muFace, muNonFace, varFace, varNonFace

	def plot(self, mean, covariance, pca_components, pca_mean):
		img = np.array(np.dot(mean, pca_components) + pca_mean).astype('uint8')
		img = np.reshape(img,(20,20))


		plt.figure(figsize = (2,2))
		plt.imshow(img,cmap="gray")
		plt.figure(figsize = (2,2))
		plt.imshow(covariance)
		plt.show()

##########
#### GMM model
##########

class mixtureOfGaussians():
	def __init__(self, mean, covariance, K, weights, posterior):
		self.mean = mean
		self.covariance = covariance
		self.K = K
		self.weights = weights
		self.posterior = posterior

	# log pdf since the expoential pdf goes into overflow
	def pdf(self, i, k, X):
		v1 = np.matmul((X[:,i].reshape(-1,1) - self.mean[k]).T, inv(self.covariance[k]))
		v2 = -0.5 * np.matmul(v1, (X[:,i].reshape(-1,1) - self.mean[k]))
		prob = v2 - np.log(np.sqrt(det(self.covariance[k])*(2*np.pi**X.shape[0])))

		return self.weights[k]*prob

	# Call loop only wrt data ==> not wrt k
	def applyEM(self, i, X):
		# E-Step
		expect = 0
		for k in range(self.K):
			expect += self.weights[k]*self.pdf(i,k,X)	

		for k in range(self.K):
			self.posterior[i,k] = (self.weights[k]*self.pdf(i,k,X))/expect

		# M-Step
		for k in range(self.K):
			# Update weight
			num = 0
			den = 0
			for i in range(1000):
				num += self.posterior[i,k]
				for j in range(self.K):
					den += self.posterior[i,j]

			self.weights[k] = 1.*(num/den)

			# Update mean
			num = 0
			den = 0
			for i in range(1000):
				num += self.posterior[i,k]*X[:,i].reshape(-1,1)
				den += self.posterior[i,k]

			self.mean[k] = 1.*(num/den)

			# Update Covariance
			num = 0
			for i in range(1000):
				x = X[:,i].reshape(-1,1) - self.mean[k]
				num += self.posterior[i,k]*np.matmul(x, x.T)

			self.covariance[k] = 1.*(num/den)

	def plot(self, pca_components, pca_mean):
		img = np.array(np.dot(self.mean[0][:,0], pca_components) + pca_mean).astype('uint8')
		img = np.reshape(img,(20,20))

		plt.figure(figsize = (2,2))
		plt.imshow(img,cmap="gray")
		plt.figure(figsize = (2,2))
		plt.imshow(self.covariance[0])
		plt.show()