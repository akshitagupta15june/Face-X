  
from main import fetch_dataset, fetch_data_details, split_data, dimensionality_reduction_ICA, train_text_transform_Model, classification_svc, prediction, print_report, plot_images, title


# Load data
dataset = fetch_dataset()

# get dataset details and target names
n_samples, height, width, X, n_features, y, target_names, n_classes = fetch_data_details(dataset)

# split into a training and testing set
X_train, X_test, y_train, y_test = split_data(X, y)

# compute ICA
n_components = 150

ica, eigenfaces = dimensionality_reduction_ICA(n_components, X_train, height, width)

X_train_ica, X_test_ica = train_text_transform_Model(ica, X_train, X_test)

# Training a SVM classification model
clf = classification_svc(X_train_ica, y_train)

# Quantitative evaluation of the model quality on the test set
y_pred = prediction(clf, X_test_ica)

# printing classification report
print_report(y_test, y_pred, target_names, n_classes)


# printing images
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_images(X_test, prediction_titles, height, width)

# plot eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_images(eigenfaces, eigenface_titles, height, width)
