#"!/usr/bin/python3

import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.svm import SVC

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


###############################################

n_folds = 5
classifiers = {}
rand_state = np.random.RandomState(32)
scoring_function_label = 'f1'
scoring_function = f1_score

data = pd.read_csv('titanic_train.csv')
X = data.iloc[:,2:]
y = data.iloc[:,1]
features = list(X.columns.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)


####################################  SVM  #########################################
svm_rbf_grid = [{'gamma': np.linspace(1e-2, 1e-3,num=10), 'C': list(range(1,500,50))}]
svm_rbf_gs = GridSearchCV(SVC(random_state=rand_state, kernel='rbf'), svm_rbf_grid,
                       cv=n_folds, scoring=scoring_function_label)
svm_rbf_gs.fit(X_train, y_train)

# save the best for further analysis
classifiers['svm_rbf'] = svm_rbf_gs.best_estimator_
print("--------------------------------------------------")

print(svm_rbf_gs.cv_results_['params'])
print("Best parameters set found on development set:")
print()
print(svm_rbf_gs.best_params_)
print()
print("Grid scores on development set:")

print("--------------------------------------------------")

means = svm_rbf_gs.cv_results_['mean_test_score']
stds = svm_rbf_gs.cv_results_['std_test_score']
gamma = svm_rbf_gs.cv_results_['param_gamma']
cval = svm_rbf_gs.cv_results_['param_C']
for mean, std, params in zip(means, stds, svm_rbf_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


print("--------------------------------------------------")
parameters = {
    'gamma': (150, 155, 160),
    'C': (1, 2, 3)
}

print('Best score: %0.3f' % svm_rbf_gs.best_score_)
print('Best parameters set:')
best_parameters = svm_rbf_gs.best_estimator_.get_params()
print(best_parameters)
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))


print("--------------------------------------------------")

#plt.figure(figsize=(10, 5))
#plt.plot(neighbors,means)
#plt.xlabel('neighbors', fontsize=16)
#plt.ylabel('F1 score', fontsize=16)
#plt.savefig("knn_CV_Fscoresi.png")
#plt.close()

#score_df = pd.DataFrame({'Depth': depths,'Score': means})

score_df = pd.DataFrame(columns=svm_rbf_grid[0]['gamma'], index=svm_rbf_grid[0]['C'])
print(score_df)
print("--------------------------------------------------")
#for params, mean_score, scores in svm_rbf_gs.grid_scores_:
print(svm_rbf_gs.cv_results_['mean_test_score'])
print("------------")
#print(svm_rbf_gs.cv_results_['params'])
#print(svm_rbf_gs.cv_results_['params']['gamma'])
#for i in svm_rbf_gs.cv_results_['params']:
#    print(i['C'])

print(svm_rbf_gs.cv_results_['params'])

for m, c,g in zip(means, cval, gamma):
    score_df.loc[c][g] = m

print(score_df)

print("--------------------------------------------------")
#for params, mean_test_score in svm_rbf_gs.cv_results_:
##    score_df.loc[params['C']][params['gamma']] = mean_test_score


ax = sns.heatmap(score_df.fillna(value=0.), cmap="jet")
ax.set_title('RBF Kernel SVM C & Gamma Optimization ')
ax.set_xlabel("Gamma")
ax.set_ylabel('C')
plt.savefig("titanic-SVM-rbf-C_gammma-optimizaton.png")
#plt.show()
plt.close()


print("--------------------------------------------------")
#print('Best parameters set:')
#best_parameters = svm_rbf_gs.best_estimator_.get_params()

best_C,best_gamma=svm_rbf_gs.best_params_['C'], svm_rbf_gs.best_params_['gamma']

#print(best_C, best_gamma)

clf = svm.SVC(C=best_C, kernel="rbf", gamma=best_gamma, probability=True)
#clf = svm.SVC(C=51, kernel="rbf", gamma=0.01, probability=False, cache_size=200)
#svmrbfplot=plot_learning_curve(clf, "SVM with RBF kernel, gamma=" + str(best_gamma), X, y, ylim=[0,1])
#svmrbfplot.savefig("SVMRbfLearningCurve.png")

#plt.close()
#sys.exit(0)

print("--------------------------------------------------")
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
acc_svm_rbf = round( accuracy_score(y_test, test_predict) * 100, 2)

scores = cross_val_score(clf, X_train, y_train, cv=10, scoring = "accuracy")

print("The prediction accuracy of SVM with RBF kerel is " + str(accuracy_score(y_test, test_predict)))
print("The prediction accuracy of acc_svm_rbf is " + str(acc_svm_rbf))


print("--------------------------------------------------")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

print("F1 score:", f1_score(y_test, test_predict))
print("Precision:", precision_score(y_test, test_predict))
print("Recall:",recall_score(y_test, test_predict))

print("--------------------------------------------------")
predictions = cross_val_predict(clf, X_train, y_train, cv=10)
print("confusion_matrix:",confusion_matrix(y_test, test_predict))
cfm=confusion_matrix(y_test, test_predict)
sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None, fmt='g')
plt.title('SVM rbf kernel Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label');
#plt.show()
plt.savefig("titanic-SVM-rbf-confusionMatrix.png")
plt.close()
print("SVM rbf confusion_matrix:", cfm)

print("--------------------------------------------------")
print("SVM  Classifier with RBF kernelreport \n", classification_report(y_test, test_predict))
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
plt.savefig("titanic-SVM-rbf-roc_curve.png")
plt.close()


print("--------------------------------------------------")
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("SVM rbf ROC-AUC-Score:", r_a_score)


costs = np.power(10.0, range(-2,2))
kernels = ['linear', 'sigmoid', 'rbf']
auc_rad_tst = np.zeros((len(costs),len(kernels)))
auc_rad_trn = np.zeros((len(costs),len(kernels)))
# Comment out second layer for run time.
for i in range(len(costs)):
    for k in range(len(kernels)):
        svc = svm.SVC(kernel = kernels[k], C=costs[i], probability=True, gamma=0.01)
        svc.fit(X_train,y_train)
        auc_rad_tst[i,k] = roc_auc_score(y_test, svc.predict_proba(X_test)[:,1])
        auc_rad_trn[i,k] = roc_auc_score(y_train, svc.predict_proba(X_train)[:,1])

for k in range(len(kernels)):
    pyplot.plot(auc_rad_tst[:,k], linewidth=3, label="Test AUC  : "+kernels[k])
for k in range(len(kernels)):
    pyplot.plot(auc_rad_trn[:,k], linewidth=3, label="Train AUC : "+kernels[k])
#pyplot.grid()
plt.legend(loc='lower right')
#pyplot.ylim(0.75, 1.0)
np.set_printoptions(precision=3)
plt.xticks(range(len(costs)),['0.001', '0.01','0.1', '1','10', '100','1000'])
pyplot.xlabel("Costs")
pyplot.ylabel("validation auc")
plt.savefig("titanic-SVM-rbf-auc.png")
#pyplot.show()
plt.close()

print("--------------------------------------------------")
print("############################## Sigmoid Kernel ####################################")

# run cross validation w/ grid search over SVM parameters
svm_sig_grid = [{'gamma': [round(i,5) for i in np.linspace(1e-2, 2e-3,num=10)],
                 'coef0': [round(i,5) for i in np.linspace(0.0, 0.1,num=10)]}]

svm_sig_gs = GridSearchCV(SVC(random_state=rand_state, kernel='sigmoid'), svm_sig_grid,
                       cv=n_folds, scoring=scoring_function_label)

svm_sig_gs.fit(X_train, y_train)

# save the best for further analysis
classifiers['svm_sigmoid'] = svm_sig_gs.best_estimator_

print(svm_sig_gs.cv_results_['params'])
print("Best parameters set found on development set:")
print()
print(svm_sig_gs.best_params_)
print()
print("Grid scores on development set:")

print("--------------------------------------------------")

means = svm_sig_gs.cv_results_['mean_test_score']
stds = svm_sig_gs.cv_results_['std_test_score']
gamma = svm_sig_gs.cv_results_['param_gamma']
cval = svm_sig_gs.cv_results_['param_coef0']
for mean, std, params in zip(means, stds, svm_sig_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
parameters = {
    'gamma': (150, 155, 160),
    'coef0': (1, 2, 3)
}

print("--------------------------------------------------")
print('Best score: %0.3f' % svm_sig_gs.best_score_)
print('Best parameters set:')
best_parameters = svm_sig_gs.best_estimator_.get_params()
print(best_parameters)
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

print("--------------------------------------------------")
#score_df = pd.DataFrame(columns=svm_sig_grid[0]['gamma'], index=svm_sig_grid[0]['C'])
score_df = pd.DataFrame(columns=svm_sig_grid[0]['gamma'], index=svm_sig_grid[0]['coef0'])

print(score_df)
#for params, mean_score, scores in svm_sig_gs.grid_scores_:
print(svm_sig_gs.cv_results_['mean_test_score'])
print("------------")
#print(svm_sig_gs.cv_results_['params'])
#print(svm_sig_gs.cv_results_['params']['gamma'])
#for i in svm_sig_gs.cv_results_['params']:
#    print(i['C'])

print(svm_sig_gs.cv_results_['params'])

print("--------------------------------------------------")
for m, c,g in zip(means, cval, gamma):
    score_df.loc[c][g] = m

print(score_df)

print("--------------------------------------------------")
ax = sns.heatmap(score_df.fillna(value=0.), cmap="jet")
ax.set_title('Sigmoid Kernel SVM coeff0 & Gamma Optimization ')
ax.set_xlabel("Gamma")
ax.set_ylabel('C')
#plt.show()
plt.savefig("titanic-SVM-sigmoid-C_gammma-optimizaton.png")
plt.close()



best_coef0,best_gamma=svm_sig_gs.best_params_['coef0'], svm_sig_gs.best_params_['gamma']

#print(best_C, best_gamma)

clf = svm.SVC(coef0=best_coef0, kernel="sigmoid", gamma=best_gamma, probability=True)
#svmsigplot=plot_learning_curve(clf, "SVM with sigmoid kernel, gamma=" + str(best_gamma), X, y, ylim=[0,1])
#svmsigplot.savefig("SVMSigmoidLearningCurve.png")

plt.close()


clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
acc_svm_sig = round( accuracy_score(y_test, test_predict) * 100, 2)

scores = cross_val_score(clf, X_train, y_train, cv=10, scoring = "accuracy")

print("--------------------------------------------------")
print("The prediction accuracy of SVM with sigmoid kerel is " + str(accuracy_score(y_test, test_predict)))
print("The prediction accuracy of acc_svm_sig is " + str(acc_svm_sig))

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
print("confusion_matrix:",confusion_matrix(y_test, test_predict))
print("confusion_matrix:",confusion_matrix(y_test, test_predict))
cfm=confusion_matrix(y_test, test_predict)
sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None, fmt='g')
plt.title('SVM Classifier isigmoid confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label');
#plt.show()
plt.savefig("titanic-SVM-sigmoid-confusionMatrix.png")
plt.close()
print("SVM Sigmoid confusion_matrix:", cfm)

print("--------------------------------------------------")
print("SVM  Classifier with Sigmoid kernelreport \n", classification_report(y_test, test_predict))
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
plt.savefig("titanic-SVM-sigmoid_sig_roc_curve.png")
plt.close()


print("--------------------------------------------------")
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score Sigmoid:", r_a_score)

print("--------------------------------------------------")

