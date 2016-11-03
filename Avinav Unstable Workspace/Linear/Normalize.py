import numpy as np
import csv

with open('dataset.csv', 'r') as f:
    reader = csv.reader(f)
    X = list(reader)
X = np.delete(X, (0), axis=0)
X = np.delete(X, (0), axis=1)
X = np.delete(X, (-1), axis=1)
X = np.array(X).astype(np.float)

def normalize(X):
    #Normalize the values column-wise
    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(1, n_c):
        m = np.mean(X[:, i])
        s = np.std(X[:, i])

        mean_r.append(m)
        std_r.append(s)
        if s==0:
            s=X[0, i]
        if s == 0:
            s = float("inf")
        X_norm[:, i] = (X_norm[:, i] - m) / s
    np.savetxt("normalizedData.csv", X_norm, delimiter=",")
    print "Saved"
    #return X_norm, mean_r, std_r

normalize(X)