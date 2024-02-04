import os
import pandas as pd
import numpy as np
import random
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import sys
from sklearn.neighbors import LocalOutlierFactor
from PAU.PAU_Clustering import PAU_Clustering
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

from collections import OrderedDict
openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


def MemTest(algo, mode, system):
    instances_to = 10000000000
    if system == "M2":
        if algo == "AP":
            instances_to = 84000
        if algo == "SC":
            instances_to = 79000
        elif algo == "HAC":
            instances_to = 120000 ###
            
        folderpath = '/Users/muyeedahmed/Desktop/Research/Dataset/'
    elif system == "Jimmy":
        if algo == "AP":
            instances_to = 170000
        if algo == "SC":
            instances_to = 158000
        elif algo == "HAC":
            instances_to = 0 ###
            
        folderpath = '/jimmy/hdd/ma234/Dataset/'
        new_home_directory = '/jimmy/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    elif system == "Louise":
        if algo == "AP":
            instances_to = 80000
        if algo == "SC":
            instances_to = 75000
        elif algo == "HAC":
            instances_to = 157000
            
        folderpath = '/louise/hdd/ma234/Dataset/'
        new_home_directory = '/louise/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    elif system == "3070":
        folderpath = '../Datasets/'
    elif system == "Thelma":
        if algo == "AP":
            instances_to = 110000
        if algo == "SC":
            instances_to = 103000
        elif algo == "HAC":
            instances_to = 220000
        
        folderpath = ""
        new_home_directory = '/thelma/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    else:
        print("System name doesn't exist")
        return
    done_files = []
    if os.path.exists("Stats/" + algo + "/"+ system + "_lrd.csv"):
        done_files = pd.read_csv("Stats/" + algo + "/"+ system + "_lrd.csv")
        done_files = done_files["Filename"].to_numpy()
    else:
        if os.path.isdir("Stats/" + algo + "/") == 0:    
            os.mkdir("Stats/" + algo + "/")
        f=open("Stats/" + algo + "/"+ system + "_lrd.csv", "w")
        f.write('Filename,max,min,mean,std\n')
        f.close()
    
    files_needed_to_run = []
    if os.path.exists("MemoryStats/Time_" + algo + "_Default_" + system + ".csv"):
        dfm = pd.read_csv("MemoryStats/Time_" + algo + "_Default_" + system + ".csv")
        dfm = dfm[dfm["Completed"]==-22]
        files_needed_to_run = dfm["Filename"].to_numpy()
    #     print(files_needed_to_run)
    # return
    dataset_list = openml.datasets.list_datasets()
    
    instances_from = 50000
    
    for key, ddf in dataset_list.items():
        if "NumberOfInstances" in ddf:
            if ddf["NumberOfInstances"] >= instances_from and ddf["NumberOfInstances"] <= 100000000:
            # if ddf["NumberOfInstances"] >= instances_from:      
                """
                Kill previous process
                """
                # while True:
                #     p_name, p_id, mem = get_max_pid()
                #     if mem > 100000:
                #         command = "kill -9 " + str(p_id)
                #         os.system(command)
                #     else:
                #         break
                filename = ddf["name"]+"_OpenML" 
                filename = filename.replace(",", "_COMMA_")
                if filename in done_files:
                    print("Already done: ", filename)
                    continue
                if filename not in files_needed_to_run:
                    print("Didn't run the algo")
                    continue
                print(ddf["name"])
                id_ =  ddf["did"]
                
                try:
                    dataset = openml.datasets.get_dataset(id_)
                    
                    X, y, categorical_indicator, attribute_names = dataset.get_data(
                        dataset_format="array", target=dataset.default_target_attribute
                        )
                    df = pd.DataFrame(X)
                    df["class"] = y
                    is_numeric = df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
                except:
                    print("Failed to read data: ", id_)
                    continue
                if all(is_numeric):                
                    r = df.shape[0]
                    c = df.shape[1]
                    if "target" in df.columns:
                        y=df["target"].to_numpy()
                        X=df.drop("target", axis=1)
                    elif "class" in df.columns:
                        y=df["class"].to_numpy()
                        X=df.drop("class", axis=1)
                    else:
                        gt_available = False
                        y = [0]*r
                        X = df
                    X.fillna(X.mean(numeric_only=True).round(1), inplace=True)
                    if c < 10:
                        continue
                    try:
                        clf = LocalOutlierFactor(n_neighbors=5).fit(X)
                    
                        lrd = clf.negative_outlier_factor_
                        f=open("Stats/" + algo + "/"+ system + "_lrd.csv", "a")
                        f.write(filename+','+str(max(lrd))+','+str(min(lrd))+','+str(np.mean(lrd))+','+str(np.std(lrd))+'\n')
                        f.close()
                        
                        df = pd.DataFrame(lrd)
                        df.to_csv("lrd/"+filename+".csv", index=False, header=False)
                    except Exception as e:
                        print("Failed: ", e)
                    
               
algo = "DBSCAN"
mode = "Default"
system = sys.argv[1]

MemTest(algo, mode, system)
# MemTest("DBSCAN", "Default", "M1")


