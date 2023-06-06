#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:55:28 2023

@author: muyeedahmed
"""
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

def MemTest(algo, mode, system):
    if os.path.exists("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv") == 0:
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "w")
        f.write('Filename,Row,Columm,StartTime,EndTime\n')
        f.close()
    else:
        print("Path already Exists")

    if system == "M2":
        folderpath = '/Users/muyeedahmed/Desktop/Research/Dataset/'
    elif system == "Jimmy":
        folderpath = '/jimmy/hdd/ma234/Dataset/'
    else:
        print("System name doesn't exist")
        return
    
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        # master_files[i] = master_files[i].split("/")[-1].split(".")[0]
        master_files[i] = Path(master_files[i]).stem
    master_files.sort()
    if os.path.exists("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv"):
        done_files = pd.read_csv("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv")
        done_files = done_files["Filename"].to_numpy()
        master_files = [x for x in master_files if x not in done_files]

    
    remaining = len(master_files)
    for file in master_files:
        filepath = folderpath+file+".csv"
        runFile(file, filepath, algo, mode, system)
        print(remaining, end=" - ")
        remaining-=1


def runFile(file, filepath, algo, mode, system):
    # print(file)
    
    df = pd.read_csv(filepath)
    r = df.shape[0]
    c = df.shape[1]
    if r < 1000:
        print()
        return
    if "target" in df.columns:
        y=df["target"].to_numpy()
        X=df.drop("target", axis=1)
    elif "class" in df.columns:
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)
    else:
        y = [0]*r
        X = df
    
    X.fillna(X.mean(numeric_only=True).round(1), inplace=True)
    if c > 10:
        X = X.sample(n=10,axis='columns')

    
    print("Dataset size:", r,c)
            
    try:
        t0 = time.time()
        if mode == "Default":
            if algo == "DBSCAN":    
                clustering = DBSCAN(algorithm="brute").fit(X)
            elif algo == "AP":
                clustering = AffinityPropagation().fit(X)
            elif algo == "GMM":
                clustering = GaussianMixture(n_components=2).fit(X)
        else:
            clustering = SS_Clustering(algoName=algo)
            clustering.X = X
            clustering.y = y
            clustering.run()
            clustering.destroy()
        t1 = time.time()
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
        f.write(file+','+str(r)+','+str(c)+','+str(t0)+','+str(t1)+'\n')
        f.close()
    except:
        try:
            clustering.destroy()
        except:
            print()
        
        print(file, "killed")
            
            
algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

MemTest(algo, mode, system)
# MemTest("DBSCAN", "Default", "M2")
