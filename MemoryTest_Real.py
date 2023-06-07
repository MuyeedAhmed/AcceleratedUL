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
from sklearn.decomposition import PCA


def MemTest(algo, mode, system):
    if system == "M2":
        folderpath = '/Users/muyeedahmed/Desktop/Research/Dataset/'
    elif system == "Jimmy":
        folderpath = '/jimmy/hdd/ma234/Dataset/'
    elif system == "Louise":
        folderpath = '/louise/hdd/ma234/Dataset/'
    elif system == "3070":
        folderpath = '../Datasets/'
    elif system == "Thelma":
        folderpath = ""
    else:
        print("System name doesn't exist")
        return
    
    filestats = pd.read_csv("Utility&Test/Stats/FileStats.csv")
    filestats = filestats[filestats["Shape_R"] > 10000]
    filestats.sort_values(by=['Shape_R'])
    
    master_files = filestats["Filename"].to_numpy()
    
    if os.path.exists("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv"):
        done_files = pd.read_csv("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv")
        done_files = done_files["Filename"].to_numpy()
        master_files = [x for x in master_files if x not in done_files]
    else:
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "w")
        f.write('Filename,Row,Columm,StartTime,EndTime,Completed\n')
        f.close()
        
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
        c=c-1
    elif "class" in df.columns:
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)
        c=c-1
    else:
        y = [0]*r
        X = df
    X.fillna(X.mean(numeric_only=True).round(1), inplace=True)
    if c > 10:
        # X = X.sample(n=10,axis='columns')
        X = PCA(n_components=10).fit_transform(X)
    
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
        f.write(file+','+str(r)+','+str(c)+','+str(t0)+','+str(t1)+',1\n')
        f.close()
    except:
        try:
            clustering.destroy()
        except:
            print()
        t1 = time.time()
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
        f.write(file+','+str(r)+','+str(c)+','+str(t0)+','+str(t1)+',0\n')
        f.close()
        print(file, "killed")
            
            
algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

MemTest(algo, mode, system)
# MemTest("DBSCAN", "Default", "M2")
