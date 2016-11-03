import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import PCA
import csv
from sklearn.model_selection import KFold


with open('normalizedDataWithShares.csv', 'r') as f:
  reader = csv.reader(f)
  X = list(reader)


#X = np.delete(X, (0), axis=0)
X =  np.asarray(X)
Y = X[:, -1]  #last column (shares)

#X = np.delete(X, (0), axis=1) #delete url column

X = np.delete(X, (-1), axis=1) #delete last column from x

X = np.array(X).astype(np.float)
Y = np.array(Y).astype(np.float).reshape(-1, 1)

PCA_Data = PCA(X[..., 1:])
X = PCA_Data.Y
Ones = np.ones(shape=(X.shape[0], X.shape[1]+1))
Ones[:, 1:] = X
X = np.array(Ones).astype(np.float)
print X[0]



#print X
#print Y

#m= 39640   #Number of instances
m, n = X.shape  #Number of attributes


num_iters = 8000
alpha = 0.2

def normalize(X):
    '''
    Normalize the values column-wise

    '''
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

    return X_norm, mean_r, std_r

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
        print i, J_history[i, 0]
    return theta, J_history


def crossValidation(X,Y, k=5):

    kf = KFold(n_splits=k)
    error_mae = 0
    error_mrae =0
    error_pred =0
    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        print ("%f\n" %fold)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        theta, J_history = gradient_descent(X_train, Y_train, alpha, num_iters)
        error_mae += calculateMeanAbsoluteError(X_test,Y_test, theta)
        error_mrae += calculateMeanRelativeAbsoluteError(X_test, Y_test, theta)
        error_pred += calculatePred(X_test, Y_test, theta, 0.25)
    print "Average MAE %f" %(error_mae/k)
    print "Avergae MRAE%f"  %(error_mrae/k)
    print "Avergae PRED 0.25 %f" % (error_pred / k)



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
    # plt.scatter(Y_test, dot)

    totalErr = error.sum()
    # plt.show()
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

    print ("PRED ",q, " ", predErr)
    return predErr

# #normalize(X)
# theta, J_history = gradient_descent(X, Y, alpha, num_iters)
# print theta
# # print J_history
# plt.plot(J_history)
# plt.show()

crossValidation(X, Y)
