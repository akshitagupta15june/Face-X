"""Ordinary least squares and ridge regression, using random featurization.
Writes all performance results to `results.txt`
"""


import numpy as np
import time


def one_hot(y, num_classes=3):
    return np.eye(num_classes)[y]

t0 = time.time()

X_train, X_test = np.load('data/X_train.npy'), np.load('data/X_test.npy')
Y_train, Y_test = np.load('data/Y_train.npy'), np.load('data/Y_test.npy')
Y_oh_train, Y_oh_test = one_hot(Y_train), one_hot(Y_test)

t1 = time.time()
print('Finished loading data:', t1 - t0)


ols_train_accuracies = []
ols_test_accuracies = []
ridge_train_accuracies = []
ridge_test_accuracies = []


def evaluate(A, Y, w):
    Yhat = np.argmax(A.dot(w), axis=1)
    return float(np.sum(Yhat == Y)) / Y.shape[0]

ds = [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2304]

for d in ds:

    W = np.random.normal(size=(X_train.shape[1], d))
    A_train, A_test = X_train.dot(W), X_test.dot(W)

    t2 = time.time()
    print('Finished stacking data:', t2 - t1)

    ATA, ATy = np.ascontiguousarray(A_train.T.dot(A_train).astype(float)), A_train.T.dot(Y_oh_train)
    I = np.eye(ATA.shape[0])
    reg = 1e10
    w = np.linalg.inv(ATA).dot(ATy)
    w_ridge = np.linalg.inv(ATA + reg * I).dot(ATy)

    t3 = time.time()
    print('Finished solving:', t3 - t2)

    # ols
    ols_train_accuracy = evaluate(A_train, Y_train, w)
    ols_train_accuracies.append(ols_train_accuracy)
    print('(ols) Train Accuracy:', ols_train_accuracy)
    ols_test_accuracy = evaluate(A_test, Y_test, w)
    ols_test_accuracies.append(ols_test_accuracy)
    print('(ols) Test Accuracy:', ols_test_accuracy)

    # ridge
    ridge_train_accuracy = evaluate(A_train, Y_train, w_ridge)
    ridge_train_accuracies.append(ridge_train_accuracy)
    print('(ridge) Train Accuracy:', ridge_train_accuracy)
    ridge_test_accuracy = evaluate(A_test, Y_test, w_ridge)
    ridge_test_accuracies.append(ridge_test_accuracy)
    print('(ridge) Test Accuracy:', ridge_test_accuracy)

    t4 = time.time()
    print('Total time:', t4 - t0)


with open('outputs/results.txt', 'w') as f:
    f.write('ols,train,%s\n' % ' '.join(map(str, ols_train_accuracies)))
    f.write('ols,test,%s\n' % ' '.join(map(str, ols_test_accuracies)))
    f.write('ridge,train,%s\n' % ' '.join(map(str, ridge_train_accuracies)))
    f.write('ridge,test,%s' % ' '.join(map(str, ridge_test_accuracies)))