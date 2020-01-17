#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 07:07:42 2020

@author: root
"""
import numpy as np
import pandas as pd
import operator

#%%
class KNN(object):
    def __init__(self,X, Y):
        self.X = X  # data vector in pandas dataframe
        self.Y = Y  # class vector in pandas dataframe
        self.k = None  # no of neighbors will be set while predicting
        self.class_list  = list()   # holds unique  classes which is in Y
        self.vote_counter  = dict() # hodls  the vote for each class
        self.distance_table = pd.DataFrame(
                        np.zeros((self.X.shape[0], self.X.shape[0])),
                        columns = self.X.index
    )

# list the unique classes in Y        
    def Unique_class(self):
# convert pandas series to list
        yList = self.Y.values.tolist()
# fill the class_list with classes in the data   
        for item in yList:
            for item_2 in item:
                if not(item_2 in self.class_list):
                    self.class_list.append(item_2)
# initialize the vote_counter with 0 vote for each class
        for item in self.class_list:
            self.vote_counter.update({item: 0})

# Reset the counted vote after each operation
    def Reset_vote(self):
        for key in self.vote_counter:
            self.vote_counter[key] = 0
            
# vote counting machine
    def Vote(self, nearest_neighbors):  #nearest_neighbors is a list of indices which are closest to the data point
        self.Reset_vote()       # reset the previous vote count to zero
        for index in nearest_neighbors:
            value = self.Y.iloc[index] # extract that row of the dataframe
            value = value.values.tolist()[0] # convert to list and then to string
            self.vote_counter[value] += 1
                
 # calculates the Euclidean distance between two vectors
    def Distance(self, vector_1, vector_2):
        squared_distance = 0
        if not(len(vector_1) == len(vector_2)):
            return ("Vector dimension don't match")
        else:
            for i in range(len(vector_1)):
                squared_distance += (vector_1[i] - vector_2[i])**2
        
        return np.sqrt(squared_distance)
    
# populate the table that holds the distance each vector has with the other
    def Distance_table(self):
        #self.Unique_class()         # determine the different classes in the data
        for index_1 in self.X.index:
            vector_1 = self.X.loc[index_1,:]
            for index_2 in self.X.index:
                if (self.distance_table.loc[index_1, index_2] == 0):
                    if (self.distance_table.loc[index_2, index_1] == 0):
                        vector_2 = self.X.loc[index_2,:]
                        self.distance_table.loc[index_1,index_2] = self.Distance(vector_1, vector_2)
                    else:
                        self.distance_table.loc[index_1, index_2] = self.distance_table.loc[index_2, index_1]
        
            
# function that predict the class based on number of neighbors 
    def Predict_class(self, xTest, k):
        self.k = k
        yPredict = list()
        
        row_data = pd.Series()      # holds one row of the table
# Extracting one element from the xTest         
        for index in xTest:
            neighbors_list = list()
            row_data = self.distance_table.iloc[index]   # extracting the desired row from the table
            
            neighbors_list = row_data.nsmallest(k+1).index # get the nearest neighbors to the data
            self.Vote(neighbors_list)           # count classes of nearest neighbors
# count the vote and assign the maximum vot to the data
            yPredict.append(max(self.vote_counter.items(), key=operator.itemgetter(1))[0]) 
            #yPredict.at[index] = max(self.vote_counter.items(), key=operator.itemgetter(1))[0]
        return yPredict
        
        
    

