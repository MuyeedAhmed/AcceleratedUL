#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:55:28 2023

@author: muyeedahmed
"""

import pandas as pd
import numpy as np
import random
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation

from SS_Clustering import SS_Clustering

def dbs():
    f=open("Test/DBSCAN_Run.csv", "w")
    f.write('Row,Columm,StartTime,EndTime\n')
    f.close()
    
    for r in range(100000,1000001,100000):
        for c in range(10,101,10):
            t0 = time.time()
            try:
                df = pd.DataFrame([i*random.randrange(10) for i in np.random.rand(r, c)])
                print("Dataset size:", r,c)
                clustering = DBSCAN(algorithm="brute").fit(df)
            except:
                print("killed")
            t1 = time.time()
            f=open("Test/DBSCAN_Run.csv", "a")
            f.write(str(r)+','+str(c)+','+str(t0)+','+str(t1)+'\n')
            f.close()

def AP():
    f=open("Test/AP_Run.csv", "w")
    f.write('Row,Columm,StartTime,EndTime\n')
    f.close()
    
    for r in range(10000,100001,10000):
        for c in range(10,101,10):
            t0 = time.time()
            try:
                df = pd.DataFrame([i*random.randrange(10) for i in np.random.rand(r, c)])
                print("Dataset size:", r,c)
                clustering = AffinityPropagation().fit(df)
            except:
                print("killed")
            t1 = time.time()
            f=open("Test/AP_Run.csv", "a")
            f.write(str(r)+','+str(c)+','+str(t0)+','+str(t1)+'\n')
            f.close()

def DBSCAN_SS():
    f=open("Test/DBSCAN_SS_Run.csv", "w")
    f.write('Row,Columm,StartTime,EndTime\n')
    f.close()
    
    for r in range(100000,1000001,100000):
        for c in range(10,101,10):
            t0 = time.time()
            try:
                df = pd.DataFrame([i*random.randrange(10) for i in np.random.rand(r, c)])
                print("Dataset size:", r,c)
                
                params = []
                eps_min_samples = []
                dbscan_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
                
                params.append(["eps_min_samples", (0.5, 5), eps_min_samples])
                params.append(["algorithm", 'auto', dbscan_algorithm])
                
                clustering = SS_Clustering(parameters=params, algoName="DBSCAN")
                clustering.X = df
                clustering.y = [0] * r
                clustering.run()
            except:
                print("killed")
            t1 = time.time()
            f=open("Test/DBSCAN_SS_Run.csv", "a")
            f.write(str(r)+','+str(c)+','+str(t0)+','+str(t1)+'\n')
            f.close()
    
    
DBSCAN_SS()

# df = pd.DataFrame([i*random.randrange(10) for i in np.random.rand(1500000, 50)])
# #df = pd.DataFrame(np.random.rand(1000000, 45))
# print(df.head())

# c = DBSCAN(algorithm="kd_tree").fit(df)
# print(len(c.labels_))
