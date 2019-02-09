#"!/usr/bin/python3

import csv
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from plot_learning_curve import plot_learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals.six import StringIO

from sklearn.metrics import classification_report
from sklearn.metrics import log_loss, hamming_loss, mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from matplotlib import pyplot


import pydot
from sklearn import tree
import os
import time
import subprocess
import sys

###############################################

n_folds = 5
classifiers = {}
start = time.time()
rand_state = np.random.RandomState(32)
scoring_function_label = 'f1'
scoring_function = f1_score

algorithm_list = ['k-Nearest Neighbors', 'Decision Tree', 'Boosted Tree','SVM rbf', 'SVM sigmoid', 'Neural Network']
algorithm_performance = pd.DataFrame(index=algorithm_list,columns=['f1','precision','recall','accuracy'])

data = pd.read_csv('titanic_train.csv')
X = data.iloc[:,2:]
y = data.iloc[:,1]
features = list(X.columns.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)


####################################  Gradient Boost  #########################################

#Boosted DT classifier

gb_grid = [{'n_estimators' : list(range(1,100))}]
gb_gs = GridSearchCV(GradientBoostingClassifier(random_state=rand_state),
                       gb_grid, cv=n_folds, scoring=scoring_function_label)
gb_gs.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(gb_gs.best_params_)
print()
print("Grid scores on development set:")

means = gb_gs.cv_results_['mean_test_score']
stds = gb_gs.cv_results_['std_test_score']
estimators = gb_gs.cv_results_['param_n_estimators']
for mean, std, params in zip(means, stds, gb_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

parameters = {
    'n_estimators': (150, 155, 160),
    'learning_rate': (1, 2, 3),
}

print('Best score: %0.3f' % gb_gs.best_score_)
#print('scorer: ' % tree_gs.scorer_)
print('Best parameters set:')
best_parameters = gb_gs.best_estimator_.get_params()
print(best_parameters)

for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

plt.figure(figsize=(10, 5))
plt.plot(estimators,means)
plt.xlabel('Estimators', fontsize=16)
plt.ylabel('F1 score', fontsize=16)
plt.savefig("gb_CV_Fscores.png")

best_estimator=gb_gs.best_params_['n_estimators']
#clf = GradientBoostingClassifier(n_estimators=best_estimator)
clf = GradientBoostingClassifier(n_estimators=55, learning_rate=0.01, max_depth=3, random_state=0, max_leaf_nodes=5)
gbplot=plot_learning_curve(clf, "Gradient boost with n_estimators" + str(best_estimator), X, y, ylim=[0,1])
gbplot.savefig("GradientBoostLearningCurve.png")

clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
acc_gradientdboost = round( accuracy_score(y_test, test_predict) * 100, 2)

print("The prediction accuracy of Gardient Boost is " + str(accuracy_score(y_test, test_predict)))
print("The prediction accuracy of Gardient Boost is " + str(acc_gradientdboost))


clf = GradientBoostingClassifier(n_estimators=best_estimator)
clf = clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

print("F1 score:", f1_score(y_test, test_predict))
print("Precision:", precision_score(y_test, test_predict))
print("Recall:",recall_score(y_test, test_predict))

predictions = cross_val_predict(clf, X_train, y_train, cv=10)
print("confusion_matrix:",confusion_matrix(y_test, test_predict))

y_scores = clf.predict_proba(X_train)
y_scores = y_scores[:,1]

y_scores = clf.predict_proba(X_train)
y_scores = y_scores[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)


plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.savefig("gradientboost_roc_curve.png")
plt.close()


r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)

from sklearn.ensemble import GradientBoostingClassifier as gbm
original_params = {'n_estimators': 50, 'random_state': 2}
plt.figure()
for label, color, setting in [('Depth 2, lr = 1.0', 'turquoise', {'learning_rate': 1.0, 'max_depth': 2}),
                              ('Depth 4, lr = 1.0', 'cadetblue',      {'learning_rate': 1.0, 'max_depth': 4}),
                              ('Depth 6, lr = 1.0', 'blue',      {'learning_rate': 1.0, 'max_depth': 6}),
                              ('Depth 2, lr = 0.1', 'orange',    {'learning_rate': 0.1, 'max_depth': 2}),
                              ('Depth 4, lr = 0.1', 'red',    {'learning_rate': 0.1, 'max_depth': 6}),
                              ('Depth 6, lr = 0.1', 'purple',      {'learning_rate': 0.1, 'max_depth': 6})]:
    params = dict(original_params)
    params.update(setting)
    clf = gbm(**params)
    clf.fit(X_train, y_train)

    # compute test set auc
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict_proba(X_test)):
        test_deviance[i] = roc_auc_score(y_test, y_pred[:,1])
    #print test auc
    plt.plot((np.arange(test_deviance.shape[0]) + 1), test_deviance,
            '-', color=color, label=label)

plt.legend(loc='lower right')
pyplot.ylim(0.8, 1.0)
plt.xlabel('Boosting Iterations')
pyplot.ylabel("validation auc")
plt.figure(figsize=(12,12))
plt.show()

plt.close()

