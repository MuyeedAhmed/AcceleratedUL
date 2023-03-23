#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 05:55:46 2023

@author: muyeedahmed
"""

import sys
import os
import shutil
import glob
import pandas as pd
# import mat73
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import bisect 
from random import sample
import time
from sklearn.utils import shuffle
import csv
from scipy.io import arff

import matplotlib

datasetFolderDir = '../Dataset/Data_Small/'

def algoRun(filename):
    [X, y] = readData(filename)

    clf = OneClassSVM().fit(X)

    iter_ = clf.n_iter_
    f=open("Stats/OCSVM_nIter.csv", "a")
    f.write(filename+","+str(iter_)+","+str(X.shape[0])+","+str(X.shape[1])+"\n")
    f.close()
    
    
    

def readData(fileName):
    df = pd.read_csv(datasetFolderDir+fileName+".csv")
    # if df.shape[0] > 100000:
    #     return True
    
    df = shuffle(df)
    if "target" in df.columns:
        y=df["target"].to_numpy()
        X=df.drop("target", axis=1)
    elif "outlier" in df.columns:
        y=df["outlier"].to_numpy()
        X=df.drop("outlier", axis=1)
    else:
        print("Ground Truth not found")
 
    return X, y

def plot_iter():
    df = pd.read_csv("Stats/OCSVM_nIter.csv")
    
    ax1 = df.plot.scatter(x='n_iter', y='rows')
    ax2 = df.plot.scatter(x='n_iter', y='col')

    df2 = df[df["n_iter"] < 1000]
    ax3 = df2.plot.scatter(x='n_iter', y='rows')

    
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    master_files.sort()

    
    if os.path.exists("Stats/OCSVM_nIter.csv") == 0:
        f=open("Stats/OCSVM_nIter.csv", "w")
        f.write('Filename,n_iter,rows,col\n')
        f.close()
    
    
    # for file in master_files:
    #     print(file)
    #     try:
    #         algoRun(file)
    #     except:
    #         print("Fail")
            
    plot_iter()    
            