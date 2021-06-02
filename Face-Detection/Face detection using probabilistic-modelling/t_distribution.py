import numpy as np
from numpy.linalg import inv,det
from scipy.special import gamma,digamma,gammaln
import matplotlib.pyplot as plt
from scipy.optimize import fminbound

pca_dim = 23 

class tDist():
    
    def __init__(self,mu,covariance,v): 
        self.mean = mu
        self.covariance = covariance
        self.v = v
        self.E_h = np.zeros(pca_dim)
        self.E_log_h = np.zeros(pca_dim)
        self.delta = np.zeros(pca_dim)
    
    def prob(self,i,X):
        D = self.mean.shape[0]
        val1 = gamma((self.v + D)/2) / ( ((self.v * np.pi)** D/2) *np.sqrt(det(self.covariance))*gamma(self.v/2) )
        term = np.matmul((X[:,i].reshape(-1,1)-self.mean).T,inv(self.covariance) )                                  
        delta = np.matmul(term,(X[:,i].reshape(-1,1) - self.mean))
        val2 = (1 + delta/self.v)
        val = val1 * pow(val2, -(self.v+D)/2)
        return val[0,0]
    
    def apply_EM(self,X):
        D = self.mean.shape[0]
        
        # E-Step
        for i in range(0,pca_dim):
            term = np.matmul((X[:,i].reshape(-1,1)-self.mean).T , inv(self.covariance) )
            delta = np.matmul(term , (X[:,i].reshape(-1,1) - self.mean))

            self.delta[i] = delta
            self.E_h[i] = (self.v+D)/(self.v + delta)
            self.E_log_h[i] = digamma((self.v+D)/2) - np.log((self.v+delta)/2)

        #mean update
        #self.mean = (np.sum(self.E_h * X, axis=1)/np.sum(self.E_h)).reshape(D,1)
        self.mean = self.E_h*np.sum(X, axis = 1)
        self.mean = np.divide(self.mean, np.sum(self.E_h)).reshape(D,1)

        #covariance update
        num = np.zeros((D,D))
        for i in range(0,pca_dim):
            prod = np.matmul((X[:,i].reshape(-1,1) - self.mean), (X[:,i].reshape(-1,1) - self.mean).T)
            num = num + self.E_h[i]*prod

        self.covariance = num/np.sum(self.E_h)
        self.covariance = np.diag( np.diag(self.covariance) )
        
        #updating dof via argmin
        self.v = fminbound(t_cost, 0, 10, args=(self.E_h, self.E_log_h)) 

        for i in range(0,pca_dim):
            term = np.matmul( (X[:,i].reshape(-1,1)-self.mean).T , inv(self.covariance))
            self.delta[i] = np.matmul(term , (X[:,i].reshape(-1,1) - self.mean))
    
    def plot(self, pca_components, pca_mean):
        img = np.array(np.dot(self.mean[:,0], pca_components) + pca_mean).astype('uint8')
        img = np.reshape(img,(20,20))

        plt.figure(figsize = (2,2))
        plt.imshow(img,cmap="gray")
        plt.figure(figsize = (2,2))
        plt.imshow(self.covariance)
        plt.show()

def t_cost(v, e_h, e_logh):
   I = len(e_h)
   t1 = (v/2) * np.log((v/2))
   t2 = gammaln((v/2))
   finalCost = 0
   for i in range(I):
       t3 = ((v/2) - 1) * e_logh[i]
       t4 = (v/2) * e_h[i]
       finalCost = finalCost + t1 - t2 + t3 - t4
   finalCost = -finalCost
   return finalCost