#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:45:10 2021

@author: andresm1
"""

#DataTalks.Club Homework 1
#andresemorales@gmail.com

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"Question 1: What is the Numpy Version?"
np.__version__
#Version is 1.20.1
"Question 2: What is the Pandas Version?"
pd.__version__
#Version 1.2.4

"Getting the data"
#reading the csv file:
hwdata = pd.read_csv("homework1data.csv")
#reviewing the structure of the dataset:
hwdata.head

"Question 3: What is the average price of BMW cars in the data set?"
meandata = hwdata.groupby(by=["Make"]).mean().loc["BMW"]
meandata.MSRP
print(f"Average price for BMW in the dataset is {meandata.MSRP}")

"Question 4: Select a subset of cars after year 2015 (inclusive, i.e. 2015 and after). How many of them have missing values for Engine HP?"

data2015 = hwdata.loc[hwdata.Year>=2015]
nohpdata2015 = data2015["Engine HP"].isnull().sum()
print(f"There are {nohpdata2015} vehicles without Engine HP data in the dataset from 2015 or newer")

"Question 5: Calculate the average Engine HP in the dataset."
"Use the fillna method and to fill the missing values in Engine HP with the mean value from the previous step."
"Now, calcualte the average of Engine HP again."
"Has it changed?"

mean_hp_before = hwdata["Engine HP"].mean()
mean_hp_before

newhwdata = hwdata.copy()
newhwdata["Engine HP"] = hwdata["Engine HP"].fillna(mean_hp_before)
newhwdata["Engine HP"].isna().describe()
mean_hp_after = newhwdata["Engine HP"].mean()
round(mean_hp_after)
round(mean_hp_before)

print("No change in average HP between the old and new dataset as we are filling all the blanks with the average value")

"Question 6:"
"Select all the Rolls-Royce cars from the dataset."
"Select only columns Engine HP" "Engine Cylinders" "highway MPG"
"Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 7 rows)."
"Get the underlying NumPy array. Let's call it X."
"Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX."
"Invert XTX."
"What's the sum of all the elements of the result?"
#Hint: if the result is negative, re-read the task one more time"

X = hwdata.loc[hwdata.Make =="Rolls-Royce",["Engine HP","Engine Cylinders", "highway MPG"]].drop_duplicates()
np.array(X)
X = np.array(X)
XT = X.T
def vector_vector_multiplication(u,v):
    assert u.shape[0] == v.shape[0]
    n = u.shape[0]
    result = 0.0
    for i in range(n):
        result = result + u[i] * v[i]
    return result

def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]
    num_rows = U.shape[0]
    result = np.zeros(num_rows)
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i],v)
    return result

def matrix_matrix_multiplication(U, V):
    assert U.shape[0] == V.shape[1]
    num_rows = U.shape[0]
    num_cols = V.shape[1]
    result = np.zeros((num_rows,num_cols))
    
    for i in range(num_cols):
        vi = V[:,i]
        Uvi = matrix_vector_multiplication(U,vi)
        result[:,i] = Uvi
    return result

XTX = matrix_matrix_multiplication(X,XT)
XTX
XTX_Inv = np.linalg.inv(XTX)
XTX_Inv.sum()

print(f"The sum of all the elements of the result is {XTX_Inv.sum()}")

"Questions 7:"
"Create an array y with values [1000, 1100, 900, 1200, 1000, 850, 1300]."
"Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w."
"What's the value of the first element of w?."

y = np.array([1000,1100,900,1200,1000,850,1300])

a = XT.dot(XTX_Inv)
w = a.dot(y)
w[0]
print(f"The first value of w is {w[0]}")
