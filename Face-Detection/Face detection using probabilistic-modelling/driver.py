import sys
import numpy as np
from gaussian import simpleGaussian, mixtureOfGaussians
from t_distribution import tDist
from factor_analyzer import factorAnalyzer
from t_mixture import tMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

try:
	function = sys.argv[1]
except IndexError as msg:
	print("Invalid command line arguments, please type 'help' for more information")
	exit()

pca_dim = 23

def getPCA(data):
	pca = PCA(pca_dim)
	pca.fit(data)
	dataPCA = pca.transform(data)
	temp = StandardScaler()
	temp.fit(dataPCA)
	data = temp.transform(dataPCA)
	return data, pca

model = simpleGaussian()
model.obj1.createImgArray()

t_train_pca, pca_f = getPCA(model.obj1.ftr)
u_train_pca, pca_nf = getPCA(model.obj1.ntr)
t_test_pca, _ = getPCA(model.obj1.fts)
u_test_pca, _ = getPCA(model.obj1.nts)

t_train_pca, u_train_pca, t_test_pca, u_test_pca = t_train_pca.T, u_train_pca.T, t_test_pca.T, u_test_pca.T

P_f_f = np.array([])
P_nf_f = np.array([])
P_f_nf = np.array([])
P_nf_nf = np.array([])

def plotROC(post_p_f_f,post_p_nf_f,post_p_f_nf,post_p_nf_nf):
	FPR = []
	TPR = []
	for i in np.arange(0,1,0.0001):
		cor_face = np.sum(post_p_f_f >= i)
		cor_nonface = np.sum(post_p_nf_nf >= i)
		noncor_face = 100 - cor_face
		noncor_nonface = 100 - cor_nonface

		FPR.append(noncor_nonface / (noncor_nonface + cor_nonface))
		TPR.append(noncor_face / (cor_face + noncor_face))

	plt.plot(np.array(FPR), np.array(TPR))
	plt.show()


def calculatePosterior(p_f_f, p_nf_f, p_f_nf, p_nf_nf):
	post_p_f_f = p_f_f / (p_f_f + p_f_nf) 
	post_p_nf_f = p_nf_f / (p_f_f + p_nf_f) 
	post_p_f_nf = p_f_nf / (p_nf_f + p_nf_nf)
	post_p_nf_nf = p_nf_nf / (p_nf_f + p_nf_nf) 

	cor_face = np.sum(post_p_f_f >= 0.5)
	cor_nonface = np.sum(post_p_nf_nf >= 0.5)
	noncor_face = 100 - cor_face
	noncor_nonface = 100 - cor_nonface

	FPR = noncor_nonface / (noncor_nonface + cor_nonface)
	FNR = noncor_face / (cor_face + noncor_face)
	MCR = (noncor_nonface + noncor_face) / 200

	print('False Positive Rate:' + str(FPR))
	print('False Negative Rate:' + str(FNR))
	print('Miss Classification Rate:' + str(MCR))

	if(function == 'tdist' or function == 'factor'):
		plotROC(post_p_f_f, post_p_nf_f, post_p_f_nf, post_p_nf_nf)
		return;

	temp1 = p_nf_f/(p_nf_f + p_nf_nf) 
	labels = np.append(temp1, post_p_f_f)
	temp1 = np.zeros((100,1))
	temp2 = np.ones((100,1))
	truth = np.append(temp1, temp2)

	truth = np.nan_to_num(truth, copy=False)
	labels = np.nan_to_num(labels, copy=False)
	FPR, TPR, _ = roc_curve(truth, labels)
	roc_auc = auc(FPR, TPR)
	plt.plot(FPR, TPR)
	plt.show()	

##############
###### MAIN FUNCTION
##############

if function == 'simple':
	muFace, muNonFace, varFace, varNonFace = model.gaussModel(t_train_pca, u_train_pca)
	model.plot(muFace, varFace, pca_f.components_, pca_f.mean_)
	model.plot(muNonFace, varNonFace, pca_nf.components_, pca_nf.mean_)

	for i in range(100):
		P_f_f = np.append(P_f_f, model.pdf(t_test_pca, i, muFace, varFace))
		P_nf_f = np.append(P_nf_f, model.pdf(u_test_pca, i, muFace, varFace))
		P_f_nf = np.append(P_f_nf, model.pdf(t_test_pca, i, muNonFace, varNonFace))
		P_nf_nf = np.append(P_nf_nf, model.pdf(u_test_pca, i, muNonFace, varNonFace))

	calculatePosterior(P_f_f, P_nf_f, P_f_nf, P_nf_nf)

elif function == 'gmm':
	K = 3
	weight = np.random.dirichlet(np.ones(K), size=1)[0]
	post = np.random.dirichlet(np.ones(K), size=1000)

	means_f = np.zeros((K,pca_dim,1))
	covariances_f = np.array([np.random.uniform(low=0.0, high=1.0, size=(pca_dim,pca_dim)) * np.eye(pca_dim) for k in range(K)])
	gmm_f = mixtureOfGaussians(means_f ,covariances_f, K, weight, post)

	means_nf = np.zeros((K,pca_dim,1))
	covariances_nf = np.array([np.random.uniform(low=0.0, high=1.0, size=(pca_dim,pca_dim)) * np.eye(pca_dim) for k in range(K)])
	gmm_nf = mixtureOfGaussians(means_nf ,covariances_nf, K, weight, post)

	for i in range(1000):
		print('Learing face',i)
		gmm_f.applyEM(i, t_train_pca)
		print('Learing non face',i)
		gmm_nf.applyEM(i, u_train_pca)

	gmm_f.plot(pca_f.components_, pca_f.mean_)
	gmm_nf.plot(pca_nf.components_, pca_nf.mean_)

	for i in range(100):
		temp1 = 0
		temp2 = 0
		temp3 = 0
		temp4 = 0
		for k in range(K):
			temp1 += gmm_f.pdf(i, k, t_test_pca)
			temp2 += gmm_f.pdf(i, k, u_test_pca)
			temp3 += gmm_nf.pdf(i, k, t_test_pca)
			temp4 += gmm_nf.pdf(i, k, u_test_pca)

		P_f_f = np.append(P_f_f, temp1)
		P_f_nf = np.append(P_f_nf, temp2)
		P_nf_f = np.append(P_nf_f, temp3)
		P_nf_nf = np.append(P_nf_nf, temp4)

	calculatePosterior(P_f_f, P_f_nf, P_nf_f, P_nf_nf)

elif function == 'tdist':
	mean_f = np.mean(t_train_pca,axis=1)
	covariance_f = np.cov(t_train_pca) * np.eye(t_train_pca.shape[0])

	mean_nf = np.mean(u_train_pca,axis=1)
	covariance_nf = np.cov(u_train_pca) * np.eye(u_train_pca.shape[0])

	#initializing t_distribution model
	tdist_f = tDist(mean_f.reshape(-1,1), covariance_f, v=10)
	tdist_nf = tDist(mean_nf.reshape(-1,1), covariance_nf, v=10)

	for i in range(pca_dim):
		tdist_f.apply_EM(t_train_pca)
		tdist_nf.apply_EM(u_train_pca)

	tdist_f.plot(pca_f.components_, pca_f.mean_)
	tdist_nf.plot(pca_nf.components_, pca_nf.mean_)

	for i in range(100):
	    P_f_f = np.append(P_f_f, tdist_f.prob(i, t_train_pca))
	    P_f_nf = np.append(P_f_nf, tdist_f.prob(i, u_test_pca))
	    P_nf_f = np.append(P_nf_f, tdist_nf.prob(i, t_test_pca))
	    P_nf_nf = np.append(P_nf_nf, tdist_nf.prob(i, u_test_pca))

	calculatePosterior(P_f_f, P_f_nf, P_nf_f, P_nf_nf)

elif function == 'factor':
	K = 10
	mean_f = np.mean(t_train_pca,axis=1)
	covariance_f = np.diag(np.diag(np.cov(t_train_pca)))
	phi_f = np.random.rand(pca_dim, K)

	mean_nf = np.mean(u_train_pca,axis=1)
	covariance_nf = np.diag(np.diag(np.cov(u_train_pca)))
	phi_nf = np.random.rand(pca_dim, K)

	#initializing t_distribution model
	factor_f = factorAnalyzer(mean_f.reshape(-1,1), covariance_f, phi_f,K)
	factor_nf = factorAnalyzer(mean_nf.reshape(-1,1), covariance_nf, phi_nf,K)

	for i in range(pca_dim):
		factor_f.applyEM(t_train_pca)
		factor_nf.applyEM(u_train_pca)

	factor_f.plot(pca_f.components_, pca_f.mean_)
	factor_nf.plot(pca_nf.components_, pca_nf.mean_)

	for i in range(100):
	    P_f_f = np.append(P_f_f, factor_f.pdf(i, t_train_pca))
	    P_f_nf = np.append(P_f_nf, factor_f.pdf(i, u_test_pca))
	    P_nf_f = np.append(P_nf_f, factor_nf.pdf(i, t_test_pca))
	    P_nf_nf = np.append(P_nf_nf, factor_nf.pdf(i, u_test_pca))

	calculatePosterior(P_f_f, P_f_nf, P_nf_f, P_nf_nf)

elif function == 'tmix':
	K = 3
	v_face = 25*np.ones((K,1))
	means_f = np.random.rand(K,pca_dim,1)
	means_nf = np.random.rand(K,pca_dim,1)

	v_nonface = 25*np.ones((K,1))
	covariances_f = np.random.rand(K,pca_dim,pca_dim)
	covariances_nf = np.random.rand(K,pca_dim,pca_dim)

	tMix_f = tMixture(means_f, covariances_f, v_face, K)
	tMix_nf = tMixture(means_nf, covariances_nf, v_nonface, K)

	for i in range(1000):
		print('Learing face',i)
		tMix_f.applyEM(i, t_train_pca)
		print('Learing non face',i)
		tMix_nf.applyEM(i, u_train_pca)

	tMix_f.plot(pca_f.components_, pca_f.mean_)
	tMix_nf.plot(pca_nf.components_, pca_nf.mean_)

	for i in range(100):
		temp1 = 0
		temp2 = 0
		temp3 = 0
		temp4 = 0
		for k in range(K):
			temp1 += tMix_f.pdf(i, k, t_test_pca)
			temp2 += tMix_f.pdf(i, k, u_test_pca)
			temp3 += tMix_nf.pdf(i, k, t_test_pca)
			temp4 += tMix_nf.pdf(i, k, u_test_pca)

		P_f_f = np.append(P_f_f, temp1)
		P_f_nf = np.append(P_f_nf, temp2)
		P_nf_f = np.append(P_nf_f, temp3)
		P_nf_nf = np.append(P_nf_nf, temp4)

	calculatePosterior(P_f_f, P_f_nf, P_nf_f, P_nf_nf)

elif function == 'help':
	print("Enter 'simple' for executing simple gaussian model")
	print("Enter 'gmm' for mixture model")
	print("Enter 'tdist' for t-distribution model")
	print("Enter 'factor' for factor analyzer")
	print("Enter 'tmix' for mixture of t-distributions model")