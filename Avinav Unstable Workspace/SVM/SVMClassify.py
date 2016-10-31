import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import svm
from sklearn.model_selection import KFold




with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    X = list(reader)

# X = np.delete(X, (0), axis=0)
X = np.asarray(X)
Y = X[:, -1]  # last column (shares)

# X = np.delete(X, (0), axis=1) #delete url column

X = np.delete(X, (-1), axis=1)  # delete last column from x

X = np.array(X).astype(np.float)
Y = np.array(Y).astype(np.float).reshape(-1, 1)

# print X
# print Y

# m= 39640   #Number of instances
m, n = X.shape  # Number of attributes


def crossValidation(X, Y, k=10):
    # Cross Validation Coeff
    # Constants for confusion MAtrix
    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0

    kf = KFold(n_splits=k)
    error_mae = 0
    error_mrae = 0
    error_pred = 0
    fold = 1
    for train_index, test_index in kf.split(X):
        print("\n\nFold %f" % fold)
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf = svm.SVC()
        clf.fit(X_train, Y_train)
        svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        Y_calc = clf.predict(X_test)
        #        theta, J_history = gradient_descent(X_train, Y_train, alpha, num_iters)

        tp, fp, fn, tn = calculateConfusionMatrix(Y_calc, Y_test)
        true_p += tp
        true_n += tn
        false_p += fp
        false_n += fn
    # print "Average MAE %f" % (error_mae / k)
    # # print "Avergae MRAE%f" % (error_mrae / k)
    # # print "Avergae PRED 0.25 %f" % (error_pred / k)
    print "Average Confusion Matrix"
    print true_p, false_p
    print false_n, true_n
    print "Correct: ", (float)(true_n + true_p) / (float)(false_n + true_n + true_p + false_p)


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


# #normalize(X)
# theta, J_history = gradient_descent(X, Y, alpha, num_iters)
# print theta
# # print J_history
# plt.plot(J_history)
# plt.show()


crossValidation(X, Y)
