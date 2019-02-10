
Code files names are in the format <algorithm>-\<data\>.py

In order to generate the statistics and graphs for the algorithm using a dataset, run code as

python3 <algorithm>-<data>.py

Here are the commands to generate the graphs and statistics.

For Titanic Dataset (titanic_train.csv):
=============================================
python3 adaBoost-dtree-titanic.py
python3 ann-mlp-titanic.py
python3 dtree-titanic.py
python3 knn-titanic.py
pyhton3 svm-titanic.py
==========================

For Winequality dataset(winequality-data.csv)
==============================================
python3 adaBoost-dtree-wine.py
python3 ann-mlp-wine.py
python3 dtree-wine.py
python3 knn-wine.py
python3 svm-wine.py
============================

In order to generate the learning curve,
following files need to be present in directory along with above files

- plot_learning_curve.py
- titanic_train.csv
- winequality-data.csv

==================================================
