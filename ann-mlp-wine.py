#"!/usr/bin/python3

import csv
import sys
import subprocess
import numpy as np
import pandas as pd
import graphviz
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn.tree as tree
from sklearn.tree import export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from plot_learning_curve import plot_learning_curve

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

###############################################
n_folds = 5
classifiers = {}
rand_state = np.random.RandomState(32)
#scoring_function_label = 'f1_micro'
scoring_function_label = 'f1'
scoring_function = f1_score


data = pd.read_csv('winequality-data.csv')
#data = pd.read_csv('winequality-red.csv')

#bins = [1,4,6,10]
bins = [1,6,10]
#quality_labels=[0,1,2]
quality_labels=[0,1]
data['quality_categorical'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)
data=data.drop(['id'], axis =1)
quality_raw = data['quality_categorical']
features_raw = data.drop(['quality', 'quality_categorical'], axis = 1)

X = data.iloc[:,:11]
y = data.iloc[:,12]

features = list(X.columns.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)



####################################  Neural Networks #########################################
#    'hidden_layer_sizes': [(1,), (2,), (3,) ,(4,), (5,)],
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.01, 0.05, 0.1],
    'learning_rate': ['constant','adaptive'],
}
mlp = MLPClassifier(max_iter=100)
ann_gs = GridSearchCV(mlp, parameter_space, n_jobs=-1,
                      cv=n_folds, scoring='accuracy')
ann_gs.fit(X_train, y_train)

classifiers['ANN'] = ann_gs.best_estimator_
#print(classifiers['DecisionTree'])
#print(ann_gs.cv_results_)
print(ann_gs.cv_results_['params'])
print("--------------------------------------------------")
print("Best parameters set found on development set:")
print()
print(ann_gs.best_params_)
print("--------------------------------------------------")
print()
print("Grid scores on development set:")

means = ann_gs.cv_results_['mean_test_score']
stds = ann_gs.cv_results_['std_test_score']
#depths = ann_gs.cv_results_['param_max_depth']
for mean, std, params in zip(means, stds, ann_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("--------------------------------------------------")

parameters = {
    'activation': (150, 155, 160),
    'alpha': (1, 2, 3),
    'hidden_layer_sizes': (1, 2, 3),
    'learning_rate': (1, 2, 3),
    'solver': (1, 2, 3)
}

print('Best score: %0.3f' % ann_gs.best_score_)
#print('scorer: ' % ann_gs.scorer_)
print('Best parameters set:')
best_parameters = ann_gs.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))


print("--------------------------------------------------")
#score_df = pd.DataFrame({'Depth': depths,'Score': means})
#
#print(score_df)
#
#print("--------------------------------------------------")
#plt.figure(figsize=(10, 5))
#plt.plot(depths,means)
#plt.xlabel('Depth', fontsize=16)
#plt.ylabel('Accuracy score', fontsize=16)
#plt.show()
#plt.close()
#plt.savefig("DTree_CV_Fscores")

#best_depth=ann_gs.best_params_['max_depth']
#{'activation': 'logistic', 'alpha': 0.05, 'hidden_layer_sizes': (50, 100, 50), 'learning_rate': 'constant', 'solver': 'adam'}
#clf = MLPClassifier(activation = 'logistic', alpha = 0.01, hidden_layer_sizes = (50, 100, 50), learning_rate = 'constant', solver = 'adam', random_state=0)
clf = MLPClassifier(activation = 'logistic', alpha = 0.05, hidden_layer_sizes = (50, 100, 50), learning_rate = 'constant', solver = 'adam', random_state=0)
clf=clf.fit(X_train, y_train)
dplot=plot_learning_curve(clf, "MLP Neural Networks " , X, y)
#dplot.show()
dplot.savefig("wine-ann_learning_curve.png")
dplot.close()

#clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
acc_ann = round( accuracy_score(y_test, test_predict) * 100, 2)

print("--------------------------------------------------")
print("The prediction accuracy of Neural Networks  is " + str(accuracy_score(y_test, test_predict)))
print("The prediction accuracy of Neural Networks is " + str(acc_ann))
print("--------------------------------------------------")

scores = cross_val_score(clf, X_train, y_train, cv=10, scoring = "accuracy")

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
plt.title('Neural Network Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('wine-confusion_ann.png')
#plt.show()
plt.close()
#print("confusion_matrix:",confusion_matrix(y_test, test_predict))
print("confusion_matrix:", cfm)

print("--------------------------------------------------")
y_scores = clf.predict_proba(X_train)
y_scores = y_scores[:,1]
print("--------------------------------------------------")
print("Neural networke Classifier report") 
print(classification_report(y_test, test_predict))
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
#plt.show()
plt.savefig("wine-ann-roc_curve.png")
plt.close()

from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)


print("--------------------------------------------------")


#clf = MLPClassifier(activation = 'logistic', alpha = 0.01, hidden_layer_sizes = (100,), learning_rate = 'constant', solver = 'lbfgs', random_state=0)
#clf = MLPClassifier(activation = 'tanh', alpha = 0.05, hidden_layer_sizes = (50, 50, 50), learning_rate = 'constant', solver = 'adam', random_state=0)
clf = MLPClassifier(activation = 'logistic', alpha = 0.05, hidden_layer_sizes = (50, 100, 50), learning_rate = 'constant', solver = 'adam', random_state=0)
clf.epochs = 100
clf=clf.fit(X_train, y_train)
#plt.plot(range(len(clf.loss_)), clf.loss_)
plt.plot(clf.loss_curve_)
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.savefig('wine-ann-cost_curve.png')
plt.close()


