# Importing libraries
from time import time
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import warnings

# Fetching the dataset
# To change the dataset, alter the dataset geting return
def fetch_dataset():
	dataset = fetch_lfw_people(min_faces_per_person = 100)

	return dataset


def get_dataset_details(dataset):
	n_samples, height, width = dataset.images.shape

	X = dataset.data
	n_features = X.shape[1]

	# The identification label
	y = dataset.target
	t_names = dataset.target_names
	n_classes = t_names.shape[0]

	print("Dataset Size: ")
	print("n_samples: %d" %n_samples)
	print("n_features: %d" %n_features)
	print("n_classes: %d" %n_classes)
	print()

	return n_samples, height, width, X, n_features, y, t_names, n_classes


# Splitting the dataset in train and test data sets
# Change the test size here
def split_data(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

	return X_train, X_test, y_train, y_test 


# reducing the dimenionality of the dataset using PCA
def dimensionality_reduction(n_components, X_train, y_train):
	print("Selecting top %d fisher faces from %d faces" %(n_components, X_train.shape[0]))

	t_begin = time()
	
	pca = PCA(n_components = n_components).fit(X_train)
	lda = LDA().fit(pca.transform(X_train), y_train)

	print("Time taken: %0.3fs\n" %(time() - t_begin))

	return lda, pca


# Projecting the dataset on the eigenfaces orthonormal basis
def train_text_transform(lda, pca, X_train, X_test):
	print("Projecting the dataset on the eigenfaces orthonormal basis")

	t_begin = time() 
	X_train = lda.transform(pca.transform(X_train))
	X_test = lda.transform(pca.transform(X_test))

	time_taken = time() - t_begin
	print("Time taken: %0.3fs\n" %time_taken)

	return X_train, X_test

# Fitting classifier
def classification(X_train_model, y_train):
	print("Fitting classifier to the training set")

	t_begin = time()
	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
	
	clf = GridSearchCV(SVC(kernel = 'rbf', class_weight = 'balanced'), param_grid)
	clf = clf.fit(X_train_model, y_train)

	time_taken = time() - t_begin

	print("Time taken: %0.3fs" %time_taken)
	print("Best estimator selected by grid search: ")
	print(clf.best_estimator_)
	print()

	return clf


# finding y_prediction
def prediction(model, data):
	print("Printing names of faces on test set")

	t_begin = time()
	y_pred = model.predict(data)
	time_taken = time() - t_begin

	print("Time taken: %0.3fs\n" %time_taken)

	return y_pred


# print classification report and confusion matrix
def report(y_test, y_pred, t_names, n_classes):
	print("Classification Report: ")
	print(classification_report(y_test, y_pred, target_names = t_names))
	print()
	print("Confusion Matrix: ")
	print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))
	print()


# Plotting the data

matplotlib.rcParams.update(
	{
		'text.usetex': False,
		# Use the Computer modern font
		'font.family': 'stixgeneral',
		'font.serif': 'cmr10',
		'mathtext.fontset': 'cm',
		# Use ASCII minus
		'axes.unicode_minus': False,
	}
)

def plot_images(images, titles, height, width, n_row = 1, n_col = 4):
	plt.figure(figsize=(1.8*n_col, 2.6*n_row))
	plt.subplots_adjust(bottom = 0, left = 0.01, right = 0.99, top = 0.90, hspace = 0.35)

	for i in range(n_row*n_col):
		plt.subplot(n_row, n_col, i + 1)
		plt.imshow(images[i].reshape((height, width)), cmap = plt.cm.gray)
		plt.title(titles[i], size = 12)
		plt.xticks(())
		plt.yticks(())

	warnings.filterwarnings("ignore", message="Glyph 9 missing from current font.")
	plt.show()



def plot_with_pca(pca, lda, titles, height, width, n_row = 1, n_col = 4):
	plt.figure(figsize = (1.8*n_col, 2.4*n_row))
	for i in range(n_row * n_col):
		plt.subplot(n_row, n_col, i + 1)
		plt.imshow(pca.inverse_transform(lda.scalings_[:, i]).reshape((height, width)), cmap = plt.cm.gray)
		plt.title(titles[i], size = 12)
		plt.xticks(())
		plt.yticks(())

	plt.show()


# The predicted titles
def titles(y_pred, y_test, t_names, i):
	pred_name = t_names[y_pred[i]].rsplit(' ', 1)[-1]
	real_name = t_names[y_test[i]].rsplit(' ', 1)[-1]
	
	return 'predicted: %s\n true: 	%s' %(pred_name, real_name)





# Main
# Loading dataset
dataset = fetch_dataset()

# get dataset details and target names
n_samples, height, width, X, n_features, y, t_names, n_classes = get_dataset_details(dataset)

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Computing Linear Discriminant Analysis
n_components = 150 
lda, pca = dimensionality_reduction(n_components, X_train, y_train)
X_train_lda, X_test_lda = train_text_transform(lda, pca, X_train, X_test)


# training a SVM classification model
clf = classification(X_train_lda, y_train)

# Quantitative evaluation of the model quality on the test set
y_pred = prediction(clf, X_test_lda)

# printing report
report(y_test, y_pred, t_names, n_classes)

# print assets
prediction_titles = [titles(y_pred, y_test, t_names, i) for i in range(y_pred.shape[0])]

plot_images(X_test, prediction_titles, height, width)

# plot fisherfaces
fisherfaces_names = ["fisherface %d" % i for i in range(4)]
plot_with_pca(pca, lda, fisherfaces_names, height, width)




 