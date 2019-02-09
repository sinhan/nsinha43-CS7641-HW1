#"!/usr/bin/python3

import csv
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from plot_learning_curve import plot_learning_curve

from sklearn.metrics import classification_report
from sklearn.metrics import log_loss, hamming_loss, mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix



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

####################################  KNN  #########################################
knn_param_grid = [{'n_neighbors' : list(range(1,100))}]
                               
knn_gs = GridSearchCV(KNeighborsClassifier(), knn_param_grid,
                       cv=n_folds, scoring=scoring_function_label)

knn_gs.fit(X_train, y_train)
classifiers['KNN'] = knn_gs.best_estimator_

print("--------------------------------------------------")
print("Best parameters set found on development set:")
print()
print(knn_gs.best_params_)
print()
print("Grid scores on development set:")


means = knn_gs.cv_results_['mean_test_score']
stds = knn_gs.cv_results_['std_test_score']
neighbors = knn_gs.cv_results_['param_n_neighbors']
for mean, std, params in zip(means, stds, knn_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


parameters = {
    'n_neighbors': (150, 155, 160),
    'leaf_size': (1, 2, 3),
    'weights': (1, 2, 3)
}

print("--------------------------------------------------")

print('Best score: %0.3f' % knn_gs.best_score_)
print("--------------------------------------------------")
print('Best parameters set:')
best_parameters = knn_gs.best_estimator_.get_params()
print(best_parameters)
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

print("--------------------------------------------------")

plt.figure(figsize=(10, 5))
plt.plot(neighbors,means)
plt.xlabel('neighbors', fontsize=16)
plt.ylabel('F1 score', fontsize=16)
#plt.show()
plt.savefig("mushrooms-KNN-CV_Fscores.png")
plt.close()

best_nn=knn_gs.best_params_['n_neighbors']
clf = KNeighborsClassifier(n_neighbors=best_nn, weights="uniform", p=2, leaf_size=30,  algorithm='auto', metric='minkowski')
knnplot=plot_learning_curve(clf, "KNN  with nearest neighbours " + str(best_nn), X, y)
#knnplot.show()
knnplot.savefig("mushrooms-KNN-LearningCurve.png")
plt.close()


clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
acc_knn = round( accuracy_score(y_test, test_predict) * 100, 2)

print("--------------------------------------------------")

print("The prediction accuracy of KNN is " + str(accuracy_score(y_test, test_predict)))
print("The prediction accuracy of KNN is " + str(acc_knn))

#clf = KNeighborsClassifier(n_neighbors=best_nn, weights="uniform", p=2, leaf_size=30,  algorithm='auto', metric='minkowski')
#clf = clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring = "accuracy")

print("--------------------------------------------------")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

print("--------------------------------------------------")
print("F1 score:", f1_score(y_test, test_predict))
print("Precision:", precision_score(y_test, test_predict))
print("Recall:",recall_score(y_test, test_predict))

print("--------------------------------------------------")
predictions = cross_val_predict(clf, X_train, y_train, cv=10)

cfm=confusion_matrix(y_test, test_predict)
sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None, fmt='g')
plt.title('KNN Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label');
#plt.show()
plt.savefig("mushrooms-KNN-confusionMatrix.png")
plt.close()
print("confusion_matrix:", cfm)

print("--------------------------------------------------")
print("KNN Classifier report \n", classification_report(y_test, test_predict))
print("--------------------------------------------------")

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
#plt.show()
plt.savefig("mushrooms-KNN-roc_curve.png")
plt.close()

from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)

print("--------------------------------------------------")
kn = range(5,35,5)
kauc_trn, kauc_tst = np.zeros(len(kn)), np.zeros(len(kn))
for i, k in zip(range(0, len(kn)), kn):
    clf1 = KNeighborsClassifier(n_neighbors=k, algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, p=2, weights='uniform')
    clf1 = clf1.fit(X_train, y_train)
    pred_tst = clf1.predict_proba(X_test)[:,1]
    pred_trn = clf1.predict_proba(X_train)[:,1]
    kauc_tst[i] = roc_auc_score(y_test, pred_tst)
    kauc_trn[i] = roc_auc_score(y_train, pred_trn)

pyplot.plot(kn, kauc_tst, linewidth=3, label = "KNN test AUC")
pyplot.plot(kn, kauc_trn, linewidth=3, label = "KNN train AUC")
#pyplot.grid()
pyplot.legend()
pyplot.ylim(0.5, 1.0)
pyplot.xlabel("k Nearest Neighbors - Euclidean")
pyplot.ylabel("validation auc")
plt.figure(figsize=(12,12))
plt.savefig("mushrooms-KNN-validation_auc.png")
#pyplot.show()
plt.close()

