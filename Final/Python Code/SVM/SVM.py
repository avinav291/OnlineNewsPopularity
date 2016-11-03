import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import svm
from sklearn.model_selection import KFold


#	open the csv file for reading data
#	X: stores the whole csv file
with open('normalizedData.csv', 'r') as f:
    reader = csv.reader(f)
    X = list(reader)

#	dividing X into X and Y where:
#	X:	all the attributes in normailized form
#	Y:	predicted shares in 0 and 1 form
X = np.asarray(X)
Y = X[:, -1]  # last column (shares)

X = np.delete(X, (-1), axis=1)  # delete last column from x

X = np.array(X).astype(np.float)	# convert all values to float
Y = np.array(Y).astype(np.float).reshape(-1, 1)

# 	m:	Number of Instances
# 	n: 	Number of Attributes
m, n = X.shape


# Cross Validaton code that contains
# InputParameters:
# 	X:	attribute normalized variables
#	Y:	shares
#	k:	folds in cross validation by default 10
# 	Cost:	Cost for SVC classifier
# Return:	None
def SVCCrossValidation(X, Y,Cost=1.0, k=10):

    # Constants for confusion MAtrix
    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0

    # KFold library function(Sklearn) for calculating train and test indices
    kf = KFold(n_splits=k)
    fold = 1
    for train_index, test_index in kf.split(X):
        print("\n\nFold %f" % fold)
        fold += 1
        # X_train:	train attributes data matrix
        # Y_train:	train shares matrx
        # X_test:	test attributes data matrix
        # Y_test:	test shares data matrix
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        Y_train = np.array(Y_train)
        print Y_train.shape
        # SVM classifier with paramters
        # Kernel:Rbf
        clf = svm.SVC()
        clf.fit(X_train, Y_train)
        svm.SVC(C=Cost, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

        # Y_calc as the predicted Y for the test Data by the SVC Classifier
        Y_calc = clf.predict(X_test)

        # Confusion Matrix Calculation
        tp, fp, fn, tn = calculateConfusionMatrix(Y_calc, Y_test)

        true_p += tp
        true_n += tn
        false_p += fp
        false_n += fn

    print "Average Confusion Matrix"
    print true_p, false_p
    print false_n, true_n
    print "Correct: ", (float)(true_n + true_p) / (float)(false_n + true_n + true_p + false_p)

# Calculation of confusion matrix
# Input paramters:
# Y_calc:	Calculated value of share group(0/1)
# Y_test:	Actual Value of test group(0/1)
# Output Parameters:
# confusion matrix Paramters
def calculateConfusionMatrix(Y_calc, Y_test):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(0, Y_calc.shape[0]):
        if (Y_calc[i] >= 0.5):
            if (Y_test[i][0] >= 0.5):
                true_positive += 1
            else:
                false_positive += 1

        else:
            if (Y_test[i][0] >= 0.5):
                false_negative += 1
            else:
                true_negative += 1

    print "Confusion Matrix"
    print true_positive, false_positive
    print false_negative, true_negative
    print "Correct: ", (float)(true_negative + true_positive) / (float)(
        false_negative + true_negative + true_positive + false_positive)
    return true_positive, false_positive, false_negative, true_negative


SVCCrossValidation(X, Y)
