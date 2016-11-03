import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import KFold


with open('normalizedDataWithShares.csv', 'r') as f:
  reader = csv.reader(f)
  X = list(reader)

X =  np.asarray(X)
Y = X[:, -1]  #last column (shares)

X = np.delete(X, (-1), axis=1) #delete last column from x

X = np.array(X).astype(np.float)
Y = np.array(Y).astype(np.float).reshape(-1, 1)

m, n = X.shape  #Number of Instances, Attributes


num_iters = 8000
alpha = 0.2

def compute_cost(X, Y, theta):

   # No of Training Samples
    m = Y.size

    predictions = X.dot(theta)

    errors = (predictions - Y)

    J = (1.0 / (2 * m)) * errors.T.dot(errors)
    return J


def gradient_descent(X, Y, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    #Resetting Theta
    theta = np.zeros(shape=(X.shape[1], 1))
    

    m = Y.size
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = (predictions - Y) * temp

            theta[it][0] -= alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_cost(X, Y, theta)
        # print i, J_history[i, 0]
    return theta, J_history


def LinearCrossValidation(X,Y, k=10):

    kf = KFold(n_splits=k)
    k=0
    error_mae = 0
    error_mrae =0
    error_pred =0

    for train_index, test_index in kf.split(X):
        k+=1
        print "Fold", k
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        theta, J_history = gradient_descent(X_train, Y_train, alpha, num_iters)
        error_mae_fold = calculateMeanAbsoluteError(X_test, Y_test, theta)
        error_mrae_fold = calculateMeanRelativeAbsoluteError(X_test, Y_test, theta)
        error_pred_fold = calculatePred(X_test, Y_test, theta, 0.25)

        error_mae+=error_mae_fold
        error_mrae+=error_mrae_fold
        error_pred+=error_pred_fold
    print "Average Mean Absolute Error %f" % (error_mae / k)
    print "Average Mean Relative Absolute Error%f" % (error_mrae / k)
    print "Average PRED 0.25 %f" % (error_pred / k)


def calculateMeanAbsoluteError(X_test,Y_test, theta):

    dot = X_test.dot(theta)
    error = abs(Y_test - dot)
    # plt.scatter(Y_test, dot)

    totalErr = error.sum()
    # plt.show()
    print "\nMAE%f" %(totalErr/error.shape[0])
    return totalErr/error.shape[0]

def calculateMeanRelativeAbsoluteError(X_test, Y_test, theta):
    dot = X_test.dot(theta)
    error = abs(Y_test - dot)/Y_test

    totalErr = error.sum()
    print "\nMRAE:%f" %(totalErr/error.shape[0])
    return totalErr/error.shape[0]

def calculatePred(X_test,Y_test,theta,q=0.25):

    dot = X_test.dot(theta)
    error = abs((Y_test - dot)/Y_test)
    k=0

    for i in range(0,error.shape[0]):
        if(error[i][0]<=q):
            k=k+1

    predErr = float(k)/float(error.shape[0])

    print "PRED ",q, " ", predErr
    return predErr

LinearCrossValidation(X, Y)
