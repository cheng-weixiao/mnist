# three classifiers (i.e., SVM, neural network and decision tree) are 
# used to classify handwritten digits.

import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f:
    X1, X2, X3, test = pickle.load(f)

def classifier(clf, train, test):
    """return prediction for test data and training and test error"""
    X = train[:,:-1] / 255
    y = train[:,-1]
    clf.fit(X, y)
    train_p = clf.predict(X)
    test_p = clf.predict(test[:,:-1] / 255)
    train_error = 1 - accuracy_score(y_pred=train_p, y_true=y)
    test_error = 1 - accuracy_score(y_pred=test_p, y_true=test[:,-1])
    return test_p, train_error, test_error

classifiers = [SVC(), MLPClassifier(hidden_layer_sizes=(500,)), DecisionTreeClassifier()]
clf_names = ['Support Vector Machine', 'Neural Network', 'Decision Tree']
X = [X1, X2, X3]
preds = []
for i in range(3):
    pred, train_error, test_error = classifier(classifiers[i], X[i], test)
    preds.append(pred)
    print('train and test error of {0:s} are {1:.3f} and {2:.3f}, respectively'.format(clf_names[i], train_error, test_error))

def majority_voting(s):
    """take np.array as argument, return majority component if exist"""
    s_sorted = np.sort(s)
    index = len(s) // 2
    candidate = s_sorted[index]
    counts = np.count_nonzero(s == candidate)
    if counts > index:
        return candidate
    else:
        return 255

predictions = np.vstack(preds)
vote_results = np.apply_along_axis(majority_voting, axis=0, arr=predictions)

vote_error = 1 - accuracy_score(y_pred=vote_results, y_true=test[:,-1])
print('the test error for majority voting is {0:.3f}'.format(vote_error))