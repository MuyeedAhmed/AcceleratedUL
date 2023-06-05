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


def MemTest(algo, mode, system):
    if os.path.exists("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv") == 0:
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "w")
        f.write('Row,Columm,StartTime,EndTime\n')
        f.close()
    else:
        print("Path already Exists")
        
    for r in range(100000,1000001,100000):
        for c in range(10,101,10):
            t0 = time.time()
            df = pd.DataFrame([i*random.randrange(10) for i in np.random.rand(r, c)])
            print("Dataset size:", r,c)
            
            try:
                if mode == "Default":
                    if algo == "DBSCAN":    
                        clustering = DBSCAN(algorithm="brute").fit(df)
                    elif algo == "AP":
                        clustering = AffinityPropagation().fit(df)
                else:
                    clustering = SS_Clustering(algoName=algo)
                    clustering.X = df
                    clustering.y = [0] * r
                    clustering.run()
                    clustering.destroy()
                t1 = time.time()
                f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
                f.write(str(r)+','+str(c)+','+str(t0)+','+str(t1)+'\n')
                f.close()
            except:
                try:
                    clustering.destroy()
                except:
                    print()
                print("killed")
            
            
algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

MemTest(algo, mode, system)

