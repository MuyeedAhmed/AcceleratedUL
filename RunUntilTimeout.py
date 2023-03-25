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
import numpy as np
# import mat73
# from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
# from sklearn.metrics.cluster import adjusted_rand_score
from random import sample
import time
from sklearn.utils import shuffle
import multiprocessing


datasetFolderDir = '../Dataset/Data_Small/'
# datasetFolderDir = 'Temp/'


def algoRun(filename):
    print(filename)
    [X, y] = readData(filename)
    
    p = multiprocessing.Process(target=worker, args=(X,))
    p.start()
    p.join(timeout=7200)
    if p.is_alive():
        p.terminate()
    
    step = "while_loop"
    if os.path.exists("Log/while_loop.txt"):
        with open('Log/while_loop.txt', 'rb') as f:
            last_line = f.readlines()[-2].decode()
        os.remove("Log/while_loop.txt")
    else:
        step="grd"
        with open('Log/grd.txt', 'rb') as f:
            last_line = f.readlines()[-2].decode()
    os.remove("Log/grd.txt")
    
    iter_c = last_line.split("-")[0]
    
    f=open("Stats/OCSVM_Incomplete.csv", "a")
    f.write(filename+","+step+","+str(iter_c)+","+str(X.shape[0])+","+str(X.shape[1])+"\n")
    f.close()
    print(iter_c)


def worker(X):
    clf = OneClassSVM().fit(X)


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


    
if __name__ == '__main__':
    algorithm = "OCSVM"
    folderpath = datasetFolderDir
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    if os.path.exists("Stats/"+algorithm+".csv"):
        done_files = pd.read_csv("Stats/"+algorithm+".csv")
        done_files = done_files["Filename"].to_numpy()
        master_files = [x for x in master_files if x not in done_files]
    
    master_files.sort()
        
    if os.path.exists("Stats/OCSVM_Incomplete.csv") == 0:
        f=open("Stats/OCSVM_Incomplete.csv", "w")
        f.write('Filename,Step,n_iter,rows,col\n')
        f.close()
    
    # algoRun("analcatdata_challenger")
    
    for file in master_files:
        try:
            algoRun(file)
        except:
            print("Fail")
            