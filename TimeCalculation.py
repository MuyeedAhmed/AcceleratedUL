import os
import pandas as pd
import numpy as np
import random
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import sys
from SS_Clustering import SS_Clustering
from sklearn.mixture import GaussianMixture
from pathlib import Path
import glob
from sklearn.decomposition import PCA
import psutil
import multiprocessing
import threading

import subprocess
import openml
import openml.config


import matplotlib.pyplot as plt 


def TimeCalc(algo, mode, system):
    folderpath = '../Openml/'
    
    
    done_files = []
    if os.path.exists("Stats/Time/" + algo + "/"+ system + ".csv") == 0:
        if os.path.isdir("Stats/Time/" + algo + "/") == 0:    
            os.mkdir("Stats/Time/" + algo + "/")
        f=open("Stats/Time/" + algo + "/"+ system + ".csv", "w")
        f.write('Filename,Row,Columm,Estimated_Time,1000,2000,3000,6000,9000,12000,15000,20000\n')
        f.close()
    else:
        done_files = pd.read_csv("Stats/Time/" + algo + "/"+ system + ".csv")
        done_files = done_files["Filename"].to_numpy()

    
    master_files = glob.glob(folderpath+"*.csv")

    for file in master_files:
        filename = file.split("/")[-1].split(".")[0]
        if filename in done_files:
            print("Already done", filename)
            continue
        print(filename)
        df = pd.read_csv(file)
        row = df.shape[0]
        col = df.shape[1]

        if row < 50000:
            continue
        
        
        rows = [1000,2000,3000,6000,9000,12000,15000,20000]
        # rows = [300,600,900,1200,1500,1800]
        times = []
        for r in rows:
            print(r, end=' - ')
            X = df[:r]
            
            t0 = time.time()
            if algo == "AP":
                clustering = AffinityPropagation().fit(X)
            elif algo == "SC":
                clustering = SpectralClustering().fit(X)
            else:
                print("Wrong Algo")
                return
            time_ = time.time()-t0
            times.append(time_)
            print(time_)
            
        lr_func = linear_regression_function(rows, times)
        estimated_time = lr_func(row)
        
        time_str = ",".join(str(x) for x in times)
        f=open("Stats/Time/" + algo + "/"+ system + ".csv", "a")
        f.write(filename+','+str(row)+','+str(col)+','+str(estimated_time)+','+time_str+'\n')
        f.close()
        
def linear_regression_function(X, Y):
    X = np.array(X)
    Y = np.array(Y)

    m, b = np.polyfit(X, Y, deg=1)

    function = lambda x: m * x + b

    return function        
        

algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

TimeCalc(algo, mode, system)

# TimeCalc("AP", "Default", "M2")



