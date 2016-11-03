import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import svm
from sklearn.model_selection import KFold


#	open the csv file for reading data
#	X: stores the whole csv file
with open('normalizedDataWithSharesReducedAttributes.csv', 'r') as f:
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
    mae = 0

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
        clf = svm.SVC(probability=True, C=Cost)
        clf.fit(X_train, Y_train)


        # Y_calc as the predicted Y for the test Data by the SVC Classifier
        Y_calc = clf.predict(X_test)
        Y_proba = clf.predict_proba(X_test)
        Y_proba = np.delete(Y_proba, (-1), axis=1)
        print Y_proba.shape

        # Confusion Matrix Calculation
        tp, fp, fn, tn = calculateConfusionMatrix(Y_calc, Y_test)
        mae_fold=calculateMeanAbsoluteError(Y_proba, Y_test)

        mae+=mae_fold
        true_p += tp
        true_n += tn
        false_p += fp
        false_n += fn

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
    print "Recall"

def calculateMeanAbsoluteError(Y_proba,Y_test):

    error = abs(Y_test - Y_proba)
    # plt.scatter(Y_test, dot)

    totalErr = error.sum()
    # plt.show()
    print "\nMean Absolute Error%f" %(totalErr/error.shape[0])
    return totalErr/error.shape[0]







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
