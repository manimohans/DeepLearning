#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:33:12 2018

@author: mani
"""

#Data pre-processing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
#pandas is the most important lib to import data
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values #everything except first col
Y = dataset.iloc[:, -1].values #only last col

#taking care of missing data
from sklearn.preprocessing import Imputer
#sklearn is scikit learn
#imputer will help us take care of missing data

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X[:, 1:3]) #fitting 2 and 3 column - 1 and 2 since 0-index
X[:, 1:3]=imputer.transform(X[:, 1:3])
#hit ctrl+i to know more about the function used

#cannot keep text as categorical variables
#change text to numbers

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0]) #first col
#0 1 and 2 are not good forms of encoding since ML model
#might think 2>1>0 and that's not good.

#dummy encoding - create columns for each country-not each country e.g. france, not france
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#for yes/no column, labelencoder is enough
labelencoder_y = LabelEncoder()
Y=labelencoder_y.fit_transform(Y)

#splitting the data into training set and test set
#very crucial for supervised learning
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) #0.2 to 0.25 is good
#train_size+test_size=1.0
#cross_validation class is deprecated

#machine learning uses eucledian distance
#so salary needs to be scaled because the distance 
#will be mostly dominated by salary
#we need to normalize the data basically
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#for test set no need to fit

#we don't need to apply feature scaling to Y categorical values
