"""Simple least squares classifier for face emotion recognition."""

import numpy as np


def evaluate(A, Y, w):
    Yhat = np.argmax(A.dot(w), axis=1)
    return float(np.sum(Yhat == Y)) / Y.shape[0]

def main():
    # load data
    with np.load('data/fer2013_train.npz') as data:
        X_train, X_test = data['X'], data['Y']

    with np.load('data/fer2013_test.npz') as data:
        Y_train, Y_test = data['X'], data['Y']

    # one-hot labels
    I = np.eye(3)
    Y_oh_train, Y_oh_test = I[Y_train], I[Y_test]

    # select first 100 dimensions
    A_train, A_test = X_train[:, :100], X_test[:, :100]

    # train model
    w = np.linalg.inv(A_train.T.dot(A_train)).dot(A_train.T).dot(Y_oh_train)

    # evaluate model
    ols_train_accuracy = evaluate(A_train, Y_train, w)
    print('(ols) Train Accuracy:', ols_train_accuracy)
    ols_test_accuracy = evaluate(A_test, Y_test, w)
    print('(ols) Test Accuracy:', ols_test_accuracy)