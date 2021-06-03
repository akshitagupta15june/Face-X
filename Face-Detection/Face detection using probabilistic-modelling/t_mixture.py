import numpy as np
from numpy.linalg import inv,det
from scipy.special import gamma,digamma,gammaln
import matplotlib.pyplot as plt
from scipy.optimize import fmin, fminbound

pca_dim = 23 

def tCost(v, k, E_h, E_log_h):
	val = 0
	for i in range(int(E_h)):
		val += ((v[k]/2)-1)*E_log_h - (v[k]/2)*E_h - (v[k]/2)*np.log(v[k]/2) - np.log(gamma(v[k]/2))

	return -val

class tMixture():
	def __init__(self, mean, covariance, v, K):
		self.mean = mean
		self.covariance = covariance
		self.v = v
		self.K = K
		self.E_h = np.zeros((K,1000))
		self.E_log_h = np.zeros((K,1000))
		self.delta = np.zeros((K,1000))

	def pdf(self, i, k, X):
		D = self.mean[k].shape[0]
		det_sigma = det(self.covariance[k])
		if det_sigma < 0:
			det_sigma = -det_sigma

		temp1 = gamma((self.v[k]+D)/2.0)/(((self.v[k]*np.pi)**(D/2))*np.sqrt(det_sigma)*gamma(self.v[k]/2))

		## Calculate delta value
		x = np.matmul((X[:,i].reshape(-1,1)-self.mean[k]).T,inv(self.covariance[k]))
		temp2 = np.matmul(x,(X[:,i].reshape(-1,1) - self.mean[k]))
		## Delta calculation done

		temp3 = (1 + temp2/self.v[k])
		val = temp1 * pow(temp3, -(self.v[k]+D)/2)

		return val[0,0]

	# Iterate over all Ks
	def applyEM(self, i, X):
		D = self.mean.shape[1]

		# E-Step
		for k in range(self.K):
			x = X[:,i].reshape(-1,1)-self.mean[k]

			term = np.matmul(x.T, inv(self.covariance[k]))
			self.delta[k,i] = np.matmul(term, x)

			term1 = np.matmul(x.T , inv(self.covariance[k]))
			term2 = np.matmul(term1, x)[0,0]
			self.E_h[k,i] = (self.v[k]+D)/(self.v[k] + term2)

			term1 = np.matmul(x.T , inv(self.covariance[k]))
			term2 = np.matmul(term1, x)[0,0]
			self.E_log_h[k,i] = digamma((self.v[k]+D)/2)-np.log((self.v[k]+term2)/2)

    	# M-Step
    	# Updating mean
		num = np.zeros((D,1))
		den = 0
		for k in range(self.K):
			num += self.E_h[k,i]*X[:,i].reshape(-1,1)
			den += self.E_h[k,i]

		self.mean[k] = 1.*(num/den)

		num = np.zeros((D,D))
		for k in range(self.K):
			x = X[:,i].reshape(-1,1)-self.mean[k]
			num += self.E_h[k,i]*np.matmul(x,x.T)

		self.covariance[k] = np.diag(np.diag(1.*(num/den)))

    	## Optimizing using argmin
		self.v[k] = fmin(tCost, self.v, args=(k, self.E_h[k,i], self.E_log_h[k,i]))[0]

	def plot(self, pca_components, pca_mean):
		img = np.array(np.dot(self.mean[0][:,0], pca_components) + pca_mean).astype('uint8')
		img = np.reshape(img,(20,20))

		plt.figure(figsize = (2,2))
		plt.imshow(img,cmap="gray")
		plt.figure(figsize = (2,2))
		plt.imshow(np.diag(np.diag(self.covariance[0])))
		plt.show()

