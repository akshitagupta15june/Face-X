import numpy as np 
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

image_data = lfw_people.images #3D matrix (images as 2D vectors)
no_of_sample, height, width = lfw_people.images.shape
data = lfw_people.data #images as 1D arrays (flattened data)
labels = lfw_people.target #categories
target_labels = lfw_people.target_names #category names

'''Testing Image data and visualising random image'''
plt.imshow(image_data[30, :, :])
plt.show()

'''UNderstanding labels'''
print 'these are the label', labels
print 'target labels', target_labels


'''Seeing image dimensions and number of images'''
print 'number of images', no_of_sample
print 'image height and width', height, width


'''
Normalising the following matrix
'''
sk_norm = normalize(data, axis=0) 
norm_matrix = (data-np.mean(data, axis=0))/np.var(data, axis=0)

matrix = sk_norm


'''
PCA
'''
cov_matrix = matrix.T.dot(matrix)/(matrix.shape[0]) #finding the covaraince matrix
values, vectors = np.linalg.eig(cov_matrix) #Fiding the eigen vector space

red_dim = 1000
eigen_faces = vectors[:, :red_dim]

pca_vectors = matrix.dot(eigen_faces) #obtain our eigen faces

'''
Visualizing Principal Components
'''
eigen_vec = np.reshape(eigen_faces[:,1], (50,37))
plt.imshow(eigen_vec)
#plt.show()

'''
Using machine learning techniques
'''


X_train, X_test, y_train, y_test = train_test_split(pca_vectors, labels, random_state=42) #splitting test data

knn = KNeighborsClassifier(n_neighbors=10) #using a K means Classifier
knn.fit(X_train, y_train) #training data

print 'accuracy', knn.score(X_test, y_test) #applying model on test data


# cv2.imshow('original image', img)
# cv2.imshow('grey', grey_img)

