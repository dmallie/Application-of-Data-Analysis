#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:19:38 2020

@author: root
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#%% Read Data
data_path = "Applications of DataAnalysis/data_set/" + "iris.data"
iris = pd.read_table(data_path,sep=",")

iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

#%% set parameter for the algorithm
A = 10      # k range
N = 4       # number of folds
M = dict()  # the ditionary that holds the performance evaluation for each A & N
accuracy = 0

#%% CV-based performance evaluator Pcv(A, S) S == iris
# divide the dataset into N folds
split_size = np.rint(iris.shape[0]/N)

index = 0
index_list = list()

for i in range(N):
    index_list.append(index)
    index += split_size
   
#%% this function compares the accuracy of the prediction
def PerformanceEvaluator(yTest, yPred):
    global accuracy
    if not(len(yTest) == len(yPred)):
        return "Incorrect size of arrays"
    else:
        for i in range(len(yTest)):
            if yTest[i] == yPred[i]:
                accuracy += 1
        return np.rint((accuracy/len(yTest) )*100)

        
#%% Performance evaluation loop
for j in range(N):
    # take one fold as test sample while training all the others
    if j < N-1:
        Test = iris.loc[index_list[j]:index_list[j+1], :]
    else:
        Test = iris.loc[index_list[j]:,:]
    Train = iris.drop(Test.index)
    
    
    # prepare the data set for the algorithm
    xTest = Test[['sepal length', 'sepal width', 'petal length', 'petal width']]
    yTest = Test[['class']]
    yTest = np.ravel(yTest)
    
    xTrain = Train[['sepal length', 'sepal width', 'petal length', 'petal width']]
    yTrain = Train[['class']]
    yTrain = np.ravel(yTrain)
    # Hypothesis trained without fold Fj
    for k in range(A):  # Test with each number of neighbours
        # fitting KNN to the training set
        nbrs = KNeighborsClassifier(n_neighbors=k+1, algorithm='auto').fit(xTrain, yTrain)
        # predicting the class of test set
        y_pred = nbrs.predict(xTest)
        # test the performance of fold with k
        performance = PerformanceEvaluator(yTest, y_pred)
        #record the perfromance to dictionary M, j is fold while k is no of neighbors
        M.update({(j,k+1) : performance})
        # reset the value of accuracy for the next evaluation
        accuracy = 0


                
        
        
    
        