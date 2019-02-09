#"!/usr/bin/python3

import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import sklearn.tree as tree
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.preprocessing import LabelEncoder

from plot_learning_curve import plot_learning_curve

from sklearn.metrics import classification_report
from sklearn.metrics import log_loss, hamming_loss, mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


import sys

###############################################

n_folds = 5
classifiers = {}
rand_state = np.random.RandomState(32)
scoring_function_label = 'f1'
scoring_function = f1_score

data = pd.read_csv('mushrooms.csv')

labelencoder=LabelEncoder()
for column in data.columns:
    data[column] = labelencoder.fit_transform(data[column])
data=data.drop(["veil-type"],axis=1)

X=data.drop(['class'], axis=1)
y=data['class']
features = list(X.columns.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

####################################  Gradient Boost  #########################################

#Boosted DT classifier

gb_grid = [{'n_estimators' : list(range(1,100))}]
gb_gs = GridSearchCV(GradientBoostingClassifier(random_state=rand_state),
                       gb_grid, cv=n_folds, scoring=scoring_function_label)
gb_gs.fit(X_train, y_train)
print("--------------------------------------------------")
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

print("--------------------------------------------------")
parameters = {
    'n_estimators': (150, 155, 160),
    'learning_rate': (1, 2, 3),
}

print('Best score: %0.3f' % gb_gs.best_score_)
#print('scorer: ' % tree_gs.scorer_)
print('Best parameters set:')
best_parameters = gb_gs.best_estimator_.get_params()
print(best_parameters)

print("--------------------------------------------------")
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

plt.figure(figsize=(10, 5))
plt.plot(estimators,means)
plt.xlabel('Estimators', fontsize=16)
plt.ylabel('F1 score', fontsize=16)
plt.show()
plt.close()
#plt.savefig("gb_CV_Fscores.png")

print("--------------------------------------------------")
best_estimator=gb_gs.best_params_['n_estimators']
#clf = GradientBoostingClassifier(n_estimators=best_estimator)
#clf = GradientBoostingClassifier(n_estimators=58, learning_rate=0.1, max_depth=3, random_state=0, max_leaf_nodes=5)
clf = GradientBoostingClassifier(n_estimators=58, learning_rate=0.1, max_depth=3, random_state=0)
#gbplot=plot_learning_curve(clf, "Gradient boost with n_estimators" + str(best_estimator), X, y, ylim=[0,1],train_sizes=np.linspace(.1, 1.0, 10))
#plt.show()
#plt.close()
#gbplot.savefig("GradientBoostLearningCurve.png")

clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
acc_gradientdboost = round( accuracy_score(y_test, test_predict) * 100, 2)

print("The prediction accuracy of Gardient Boost is " + str(accuracy_score(y_test, test_predict)))
print("The prediction accuracy of Gardient Boost is " + str(acc_gradientdboost))

print("--------------------------------------------------")

clf = GradientBoostingClassifier(n_estimators=best_estimator)
clf = clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

print("F1 score:", f1_score(y_test, test_predict))
print("Precision:", precision_score(y_test, test_predict))
print("Recall:",recall_score(y_test, test_predict))

print("--------------------------------------------------")
predictions = cross_val_predict(clf, X_train, y_train, cv=10)

cfm=confusion_matrix(y_test, test_predict)
sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None, fmt='g')
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label');
plt.show()
plt.close()
#print("confusion_matrix:",confusion_matrix(y_test, test_predict))
print("confusion_matrix:", cfm)

print("--------------------------------------------------")
y_scores = clf.predict_proba(X_train)
y_scores = y_scores[:,1]

y_scores = clf.predict_proba(X_train)
y_scores = y_scores[:,1]
print("--------------------------------------------------")
print("Gradent Boosted Tree Classifier report \n", classification_report(y_test, test_predict))
print("--------------------------------------------------")

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)


plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
#plt.savefig("gradientboost_roc_curve.png")
plt.close()


r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)

print("--------------------------------------------------")
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
#pyplot.ylim(0.8, 1.0)
plt.xlabel('Boosting Iterations')
pyplot.ylabel("validation auc")
plt.figure(figsize=(12,12))
plt.show()
plt.close()

