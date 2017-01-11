import numpy as np
import csv
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

with open('normalizedDataWithShares.csv', 'r') as f:
  reader = csv.reader(f)
  X = list(reader)

X =  np.asarray(X)
Y = X[:, -1]  #last column (shares)
X = np.delete(X, (-1), axis=1) #delete last column from x
X = np.array(X).astype(np.float)
Y = np.array(Y).astype(np.float).reshape(-1, 1)
m, n = X.shape  #Number of Instances, Attributes

num_iters = 2000
alpha = 0.003


def hypothesis(X, theta):
    predictions = X.dot(theta)
    return 1/(1+np.e**(-1*predictions))

def cost_function(X, Y, theta):

    predictions = hypothesis(X, theta)

   # No of Training Samples
    m = Y.size

    cost = (Y).T.dot(np.log(predictions)) + (1-Y).T.dot(np.log(1-predictions))

    J = (1.0 / m) * cost
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

        predictions = hypothesis(X, theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = (predictions - Y) * temp

            theta[it][0] -= alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = cost_function(X, Y, theta)

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.suptitle("Iterations vs Cost")
    plt.plot(J_history)
    plt.show()
    return theta, J_history


def LogisticCrossValidation(X,Y, k=10):

    # Cross Validation Coeff
    # Constants for confusion MAtrix
    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0

    kf = KFold(n_splits=k)
    mae = 0
    fold = 1
    for train_index, test_index in kf.split(X):
        print("\n\nFold %f" %fold)
        fold +=1
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        theta, J_history = gradient_descent(X_train, Y_train, alpha, num_iters)

        tp, fp, fn, tn = calculateConfusionMatrix(X_test, Y_test, theta)
        mae_fold = calculateMeanAbsoluteError(X_test.dot(theta), Y_test)
        true_p+=tp
        true_n+=tn
        false_p+=fp
        false_n+=fn
        mae +=mae_fold
    print "Average Confusion Matrix"
    print true_p, false_p
    print false_n, true_n
    displayResultSummary(true_p, false_p, true_n, false_n, mae/k)

def displayResultSummary(tp, fp, tn, fn, mae):
    total = tp+fp+tn+fn
    print "Correctly Classified Instances:", (tp+tn),"\t",((float(tp+tn)/total)*100), "%"
    print "Incorrectly Classified Instances:", (fp + fn), "\t", ((float(fp+fn) / total) * 100), "%"
    print "Mean Absolute Error:\t",mae
    print "TP Rate/Recall", tp/float(tp+fn)
    print "FP Rate", fp/float(fp+tn)
    print "Precision", tp/float(tp+fp)

def calculateMeanAbsoluteError(Y_proba,Y_test):

    error = abs(Y_test - Y_proba)
    # plt.scatter(Y_test, dot)

    totalErr = error.sum()
    # plt.show()
    print "\nMean Absolute Error%f" %(totalErr/error.shape[0])
    return totalErr/error.shape[0]

def calculateConfusionMatrix(X_test,Y_test, theta):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    dot = X_test.dot(theta)

    for i in range(0, dot.shape[0]):
        if(dot[i][0]>=0.5):
            if(Y_test[i][0]>=0.5):
                 true_positive+=1
            else:
                false_positive+=1

        else:
            if(Y_test[i][0]>=0.5):
                false_negative +=1
            else:
                true_negative+=1

    print "Confusion Matrix"
    print true_positive, false_positive
    print false_negative, true_negative
    print "Correct: ", (float)(true_negative + true_positive) / (float)(false_negative + true_negative + true_positive + false_positive)
    return true_positive, false_positive, false_negative, true_negative

LogisticCrossValidation(X, Y)
