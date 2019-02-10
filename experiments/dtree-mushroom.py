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
from sklearn.tree import DecisionTreeClassifier
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


####################################  Decision Tree #########################################

tree_param_grid = [{'max_depth' : list(range(3,20))}]
tree_gs = GridSearchCV(DecisionTreeClassifier(criterion='gini',random_state=rand_state),
                       tree_param_grid, cv=n_folds, scoring=scoring_function_label)
tree_gs.fit(X_train, y_train)
classifiers['DecisionTree'] = tree_gs.best_estimator_
#print(classifiers['DecisionTree'])

#print(tree_gs.cv_results_['params'])
print("--------------------------------------------------")
print("Best parameters set found on development set:")
print()
print(tree_gs.best_params_)
print("--------------------------------------------------")
print()
print("Grid scores on development set:")

means = tree_gs.cv_results_['mean_test_score']
stds = tree_gs.cv_results_['std_test_score']
depths = tree_gs.cv_results_['param_max_depth']
for mean, std, params in zip(means, stds, tree_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("--------------------------------------------------")

parameters = {
    'max_depth': (150, 155, 160),
    'min_samples_split': (1, 2, 3),
    'min_samples_leaf': (1, 2, 3)
}

print('Best score: %0.3f' % tree_gs.best_score_)
#print('scorer: ' % tree_gs.scorer_)
print('Best parameters set:')
best_parameters = tree_gs.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))


print("--------------------------------------------------")
score_df = pd.DataFrame({'Depth': depths,'Score': means})

print(score_df)

print("--------------------------------------------------")
plt.figure(figsize=(10, 5))
plt.plot(depths,means)
plt.xlabel('Depth', fontsize=16)
plt.ylabel('F1 score', fontsize=16)
#plt.show()
plt.savefig("mushrooms-DTree-CV_Fscores.png")
plt.close()



with open('mushrooms-Dtree-tree.dot', 'w') as dotfile:
    export_graphviz(
        classifiers['DecisionTree'],
        dotfile,
        filled=True, rounded=True,proportion=True,
        feature_names=X_train.columns)
    
subprocess.call(['dot','-Tpng','mushrooms-Dtree-tree.dot','-o','mushrooms-Dtree-tree.png'])


best_depth=tree_gs.best_params_['max_depth']
clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=best_depth)
dplot=plot_learning_curve(clf, "Decision Tree with max depth " + str(best_depth), X, y)
#dplot.show()
dplot.savefig("mushrooms-DTree-LearningCurve.png")
dplot.close()

clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
acc_decision_tree = round( accuracy_score(y_test, test_predict) * 100, 2)

print("--------------------------------------------------")
print("The prediction accuracy of decision tree is " + str(accuracy_score(y_test, test_predict)))
print("The prediction accuracy of decision tree is " + str(acc_decision_tree))
print("--------------------------------------------------")

clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=best_depth)
clf = clf.fit(X_train, y_train)
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
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label');
#plt.show()
plt.savefig("mushrooms-Dtree-ConfusionMatrix.png")
plt.close()
#print("confusion_matrix:",confusion_matrix(y_test, test_predict))
print("confusion_matrix:", cfm)

print("--------------------------------------------------")
y_scores = clf.predict_proba(X_train)
y_scores = y_scores[:,1]
print("--------------------------------------------------")
print("Decision Tree Classifier report \n", classification_report(y_test, test_predict))
print("--------------------------------------------------")


features_list = X.columns.values
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(5,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.savefig("mushrooms-Dtree-featureImportance.png")
#plt.show()
plt.close()

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
plt.savefig("mushrooms-Dtree-roc_curve.png")
plt.close()

from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)

print("--------------------------------------------------")
depth = 15
tree_auc_trn, tree_auc_tst = np.zeros(depth), np.zeros(depth)
for i in range(1,depth):
    clf1 = tree.DecisionTreeClassifier(max_depth=i, criterion='gini')
    clf1 = clf1.fit(X_train, y_train)
    tree_auc_trn[i] = roc_auc_score(y_test, clf1.predict_proba(X_test)[:,1])
    tree_auc_tst[i] = roc_auc_score(y_train, clf1.predict_proba(X_train)[:,1])

#from matplotlib import pyplot
pyplot.plot(tree_auc_tst, linewidth=3, label = "Decision tree test AUC")
pyplot.plot(tree_auc_trn, linewidth=3, label = "Decision tree train AUC")
pyplot.legend()
pyplot.ylim(0.8, 1.0)
pyplot.xlabel("Max_depth")
pyplot.ylabel("validation auc")
plt.figure(figsize=(12,12))
#plt.show()
#pyplot.show()
plt.savefig("mushrooms-Dtree-Auc.png")
pyplot.savefig("mushrooms-Dtree-Auc.png")
plt.close()
