# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:13:26 2020

@author: Hari Hara Sudaan Krishnen
"""

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc
import numpy as np
#Libraries for importing functions for evaluating the performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

#importing the dataset
dataset = pd.read_csv("edu.csv")


low = "L"
medium = "M"
high = "H"
## replace elements in list.
dataset['Class'].replace(to_replace = low , value = 0,inplace = True)
dataset['Class'].replace(to_replace = medium ,value = 1,inplace = True)
dataset['Class'].replace(to_replace = high ,value = 2,inplace = True)
dataset['Class'].value_counts()


# Assigning the X and Y values
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


#encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#label encoder for categorical variables
for i in range(0, len(dataset.columns) - 1):
    z = [9, 10, 11, 12]
    if i not in z: 
        labelencoder_x = LabelEncoder()
        X[:,i] = labelencoder_x.fit_transform(X[:,i])


# One hot encoder for categorical variables - X
onehotencoder_x = OneHotEncoder(categorical_features = [1,2,3,4,5,6])
X = onehotencoder_x.fit_transform(X).toarray() 

#feature scaling - Only X
from sklearn.preprocessing import StandardScaler
standardscalar_x = StandardScaler()
X = standardscalar_x.fit_transform(X)

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 0)
from sklearn.ensemble import RandomForestClassifier
random_for = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

#without shuffling
from sklearn.model_selection import cross_val_score
print(cross_val_score(svc, X, Y, scoring = 'accuracy', cv = 10))
accuracy = cross_val_score(svc, X, Y, scoring = 'accuracy', cv = 10).mean() * 100
print("Accuracy score for SVM : ", accuracy)


from sklearn.model_selection import cross_val_score
print(cross_val_score(random_for, X, Y, scoring = 'accuracy', cv = 10))
accuracy = cross_val_score(random_for, X, Y, scoring = 'accuracy', cv = 10).mean() * 100
print("Accuracy score for Random Foreset : ", accuracy)


#Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2, random_state = 0)


svc.fit(train_X, train_Y)
svc.score(test_X, test_Y)

random_for.fit(train_X, train_Y)
random_for.score(test_X, test_Y)

#with shuffling
from sklearn.model_selection import KFold
kf = KFold(n_splits = 10, shuffle = True)
kf.get_n_splits()

from sklearn.model_selection import cross_val_score
print(cross_val_score(svc, X, Y, scoring = 'accuracy', cv = kf.get_n_splits(X)))
accuracy = cross_val_score(svc, X, Y, scoring = 'accuracy', cv = 10).mean() * 100
print("Accuracy score for SVM : ", accuracy)


from sklearn.model_selection import cross_val_score
print(cross_val_score(random_for, X, Y, scoring = 'accuracy', cv = kf.get_n_splits(X)))
accuracy = cross_val_score(random_for, X, Y, scoring = 'accuracy', cv = 10).mean() * 100
print("Accuracy score for Random Foreset : ", accuracy)