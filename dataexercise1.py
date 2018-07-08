# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:13:43 2018

@author: Jeroen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn as sk
from sklearn import preprocessing
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
import seaborn as sns


data_file = pd.read_csv('data/iris.csv')
sns.set(style="ticks")
sns.pairplot(data_file,hue="species")

le = preprocessing.LabelEncoder()

le.fit(data_file['species'])
X = data_file[data_file.columns[1:3]].values
y = le.transform(data_file['species'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33,
                                                    random_state=42)

clf = svm.SVC()
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train),"Training Score")
print(clf.score(X_test,y_test),"Testing Score")
plt.figure()
plot_decision_regions(X=X_test,y=y_test,clf=clf,legend=2)
