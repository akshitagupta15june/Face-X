import numpy as np
from numpy.linalg import det, inv
import matplotlib.pyplot as plt
from scipy.special import gamma, digamma
from gaussian import simpleGaussian

pca_dim = 23

class factorAnalyzer():
	def __init__(self, mean, covariance, phi, K):
		self.mean = mean
		self.covariance = covariance
		self.phi = phi
		self.K = K
		self.D = pca_dim
		self.E_h = np.zeros((1000,self.K, self.D))
		self.E_h_T = np.zeros((1000,self.K, self.K))

	def pdf(self, i, X):
		# Effective covariance of the distribution
		sigma = np.matmul(self.phi, self.phi.T) + self.covariance
		
		x = X[:,i].reshape(-1,1) - self.mean
		temp = np.matmul(x.T, inv(self.covariance))
		numerator = np.exp(-0.5*np.matmul(temp,x))
		det_sigma = det(sigma)
		if det_sigma < 0:
			det_sigma = -det_sigma

		return numerator/np.sqrt(det_sigma)

	def applyEM(self, X):
		# E-Step
		var11 = np.matmul(self.phi.T, inv(self.covariance))
		var1 = np.matmul(var11, self.phi) + np.eye(self.K)
		var22 = np.matmul(inv(var1),self.phi.T)
		var2 = np.matmul(var22, inv(self.covariance))
		for i in range(1000):
			self.E_h[i] = np.matmul(var2, X[:,i] - self.mean)
			self.E_h_T[i] = inv(var1) + np.matmul(self.E_h[i], self.E_h[i].T)

		# M-step
		var1 = np.zeros((pca_dim, self.K)) 
		var2 = np.zeros((self.K, self.K))
		for i in range(1000):
			var1 += np.matmul((X[:,i]-self.mean), self.E_h[i].T)
			var2 += self.E_h_T[i]
		self.phi = np.matmul(var1,inv(var2))

		var1 = np.zeros((pca_dim, pca_dim))
		for i in range(1000):
			x = X[:,i].reshape(-1,1) - self.mean
			var1 += np.matmul(x,x.T)
			var22 = np.matmul(self.phi,self.E_h[i])
			var2 = np.matmul(var22, x)
			temp = var1 - var2

		temp /= 1000
		self.covariance = np.diag(np.diag(temp))

	def plot(self, pca_components, pca_mean):
		img = np.array(np.dot(self.mean[:,0], pca_components) + pca_mean).astype('uint8')
		img = np.reshape(img,(20,20))

		plt.figure(figsize = (2,2))
		plt.imshow(img,cmap="gray")
		plt.figure(figsize = (2,2))
		plt.imshow(self.covariance)
		plt.show()