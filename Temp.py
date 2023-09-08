#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:49:50 2023

@author: muyeedahmed
"""


import pandas as pd
import time
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.cluster import adjusted_rand_score

algo = "AP"
system = "Jimmy"
mode = "Default"

folderpath = '../Openml/'

files = ["numerai28.6_OpenML", "Diabetes130US_OpenML", "BNG(vote)_OpenML", "BNG(2dplanes)_OpenML"]

for filename in files:

    df = pd.read_csv(folderpath+filename+".csv")
    print(filename, df.shape)    
    
    r = df.shape[0]
    c = df.shape[1]
    y=df["class"].to_numpy()
    X=df.drop("class", axis=1)
    c=c-1
    
    t0 = time.time()
    clustering = AffinityPropagation().fit(X)
    l = clustering.labels_
    
    ari = adjusted_rand_score(y,l)
    time_ = time.time()-t0
    print(time_)
    f=open("Stats/" + algo + "/"+ system + "_WithoutTimeLimit.csv", "a")
    f.write(filename+','+str(r)+','+str(c)+','+mode+','+system+','+str(time_)+','+str(ari)+'\n')
    f.close()