# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:52:48 2021

@author: Dell-672206
"""
import numpy as np 
import pandas as pd
import sklearn
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import LabelEncoder, scale, OneHotEncoder
from sklearn.decomposition import PCA 
from sklearn.feature_selection import RFECV 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier

import os
path = r"E:\2021SEM1\statistics for DS\Assignment\ASS1"
os.chdir(path)

#%% a) Load the data, remove null(if any), and replace all categories with values
def LoadData(): 
    pd.options.mode.chained_assignment = None
    
    # Load the data 
    df = pd.read_csv("Hitters.csv")
    
    
    # Eliminate rows with missing values(if any).
    for cname in df.columns: 
        if (df[cname].dtype == 'object'): 
            df[cname][df[cname] == '?'] = np.nan
            
    df = df.dropna()
    
    return df
def PrepareData():
    df = LoadData()
    
    df.AtBat = df.AtBat.astype('float64')
    df.Hits = df.Hits.astype('float64')
    df.HmRun = df.HmRun.astype('float64')
    df.Runs = df.Runs.astype('float64')
    df.RBI = df.RBI.astype('float64')
    df.Walks = df.Walks.astype('float64')
    df.Years = df.Years .astype('float64')
    df.CAtBat = df.CAtBat.astype('float64')
    df.CHits = df.CHits.astype('float64')
    df.CHmRun = df.CHmRun.astype('float64')
    df.CRuns = df.CRuns.astype('float64')
    df.CRBI = df.CRBI.astype('float64')
    df.CWalks = df.CWalks.astype('float64')
    df.PutOuts = df.PutOuts.astype('float64')
    df.Assists = df.Assists.astype('float64')
    df.Errors = df.Errors.astype('float64')
    df.Salary = df.Salary.astype('float64')
 
    lb_make = LabelEncoder()
    df.League = lb_make.fit_transform(df.League)
    df.Division = lb_make.fit_transform(df.Division)
    df.NewLeague = lb_make.fit_transform(df.NewLeague)
    
    df.dtypes
    
    #Get X,Y
    Y = df.Salary
    X = df.drop(["Salary"], axis = 1)
    
    return X, Y
X, Y =PrepareData()

#%% c) Fit linear regression and report 10-Fold Cross-Validation mean squared error
def Validate(X,Y, model):
    
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    mean_squ_err = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        # fit the model
        model.fit(X_train,y_train)
        
        #predict
        y_pred = model.predict(X_test)
        loss = np.mean(np.power(y_test - y_pred, 2))
        mean_squ_err.append(loss)
    return np.mean(mean_squ_err)

  
def TestModels(X, Y):
    model = LinearRegression()
    err = Validate(X, Y, model)
    print('Linear Regression  MSE= ', err) 
             
#Run Function   
X,Y = PrepareData()     
TestModels(X,Y)  
    
    