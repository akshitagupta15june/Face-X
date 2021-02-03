"""Data conversion script, from csv to numpy files.

Face Emotion Recognition dataset by Pierre-Luc Carrier and Aaron Courville

Keeps only 3 classes, where the new class indices are 0, 1, 2, respectively:
3 - happy
4 - sad
5 - surprise
"""

import argparse
import numpy as np
import csv
import time


def main():
    args = argparse.ArgumentParser('Data conversion for Face Emotion '
                                   'Recognition dataset')
    args.add_argument('--max-n-train', type=int,
                      default=np.inf,
                      help='Maximum number of training samples. Use this flag '
                           'if you run into MemoryErrors')
    args = args.parse_args()

    X = []
    Y = []

    t0 = time.time()

    with open('fer2013/fer2013.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            y = int(row[0])
            if y not in (3, 4, 5):
                continue
            y -= 3
            x = np.array(list(map(int, row[1].split())))
            X.append(x)
            Y.append(y)

    t1 = time.time()
    print('Finished loading data:', t1 - t0)

    p = 0.8
    n_val = int(len(X) * (1 - p))
    n_train = min(int(len(X) * 0.8), args.max_n_train)

    X_train, X_test = np.array(X[:n_train]), np.array(X[-n_val:])
    Y_train, Y_test = np.array(Y[:n_train]), np.array(Y[-n_val:])

    np.savez_compressed(
        'fer2013_train',
        X=X_train,
        Y=Y_train)
    print('Saved X_train %s' % str(X_train.shape))
    print('Saved Y_train %s' % str(Y_train.shape))
    np.savez_compressed(
        'fer2013_test',
        X=X_test,
        Y=Y_test)
    print('Saved X_test %s' % str(X_test.shape))
    print('Saved Y_test %s' % str(Y_test.shape))

    t2 = time.time()
    print('Finished converting data', t2 - t1)


if __name__ == '__main__':
    main()