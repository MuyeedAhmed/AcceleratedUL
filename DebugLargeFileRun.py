import sys
import os
import shutil
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
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
import threading
# from memory_profiler import profile



datasetFolderDir = '../Dataset/Data_Small/'


fileName = "analcatdata_apnea1"
df = pd.read_csv(datasetFolderDir+fileName+".csv")

if "target" in df.columns:
    y=df["target"].to_numpy()
    X=df.drop("target", axis=1)
elif "outlier" in df.columns:
    y=df["outlier"].to_numpy()
    X=df.drop("outlier", axis=1)
else:
    print("Ground Truth not found")


t0 = time.time()
c = OneClassSVM(nu=.12).fit(X)
l = c.predict(X)
l = [0 if x == 1 else 1 for x in l]

f1 = (metrics.f1_score(y, l))

t1 = time.time()
print("Default--")
print("F1: ", f1, " and Time: ", t1-t0)
    
    
        
    
        
        