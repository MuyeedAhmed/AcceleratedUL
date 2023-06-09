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


import openml
import openml.config

from collections import OrderedDict
openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


def MemTest(algo, mode, system):
    if system == "M2":
        folderpath = '/Users/muyeedahmed/Desktop/Research/Dataset/'
    elif system == "Jimmy":
        folderpath = '/jimmy/hdd/ma234/Dataset/'
        new_home_directory = '/jimmy/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    elif system == "Louise":
        folderpath = '/louise/hdd/ma234/Dataset/'
        new_home_directory = '/louise/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    elif system == "3070":
        folderpath = '../Datasets/'
    elif system == "Thelma":
        folderpath = ""
        new_home_directory = '/thelma/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    else:
        print("System name doesn't exist")
        return
    

    done_files = []
    if os.path.exists("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv"):
        done_files = pd.read_csv("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv")
        done_files = done_files["Filename"].to_numpy()
    else:
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "w")
        f.write('Filename,Row,Columm,StartTime,EndTime,Completed\n')
        f.close()
    
    dataset_list = openml.datasets.list_datasets()
    
    instances_from = 100000
    instances_to = 1000000
    
    for key, ddf in dataset_list.items():
        if "NumberOfInstances" in ddf:
            if ddf["NumberOfInstances"] >= instances_from and ddf["NumberOfInstances"] <= instances_to:
                filename = ddf["name"]+"_OpenML" 
                filename = filename.replace(",", "_COMMA_")
                if filename in done_files:
                    print("Already done: ", filename)
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
                    print("Failed to read data: ", ddf["name"], ddf["did"])
                    continue                    
                if all(is_numeric):                
                    stop_flag = threading.Event()
                    MonitorMemory = threading.Thread(target=monitor_memory_usage_pid, args=(algo, mode, system, filename,stop_flag,))
                    # RunAlgoOnDataset = multiprocessing.Process(target=runFile, args=(filename, df, algo, mode, system,))
                    
                    MonitorMemory.start()
                    # RunAlgoOnDataset.start()
                
                    # RunAlgoOnDataset.join()
                    runFile(filename, df, algo, mode, system)
                    # time.sleep(3)
                    stop_flag.set()
                    
                    # MonitorMemory.terminate()
                    MonitorMemory.join()

                    # runFile(filename, eeg, algo, mode, system)

                    if ddf["NumberOfInstances"] == instances_from:
                        instances_from += 1
                    elif ddf["NumberOfInstances"] == instances_to:
                        instances_to -= 1
                else:
                    print("In dataset ", ddf["name"], ddf["did"], "non numaric columns exists (", sum(is_numeric), "out of", len(is_numeric), ")")
    
                

def runFile(file, df, algo, mode, system):
    r = df.shape[0]
    c = df.shape[1]
    if r < 1000:
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
    try:
        if c > 10:
            # X = X.sample(n=10,axis='columns')
            columnNames = X.columns
            X = PCA(n_components=10).fit_transform(X)
            X = pd.DataFrame(X)
    except:
        print("Killed during PCA")
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

def monitor_memory_usage_pid(algo, mode, system, filename, stop_flag):
    print("Memory usage")
    interval = 0.1
    # algo = sys.argv[1]
    # mode = sys.argv[2]
    # system = sys.argv[3]
    # filename = sys.argv[4]
    
    if os.path.exists("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + ".csv") == 0:
        f=open("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + ".csv", "w")
        f.write('Name,Time,Memory_Physical,Memory_Virtual,Filename\n')
        f.close()
        
    while not stop_flag.is_set():
        memory = []
        memory_virtual = []
        name, pid = get_max_pid()
        for _ in range(10):
            try:
                process = psutil.Process(pid)
                memory_info = process.memory_info()
                memory_usage = (memory_info.rss) / (1024 * 1024)  # Convert to megabytes
                memory.append(memory_usage)
                memory_usage_virtual = memory_info.vms / (1024 * 1024)
                memory_virtual.append(memory_usage_virtual)
            except:
                print("none")
                continue
            time.sleep(interval)
        # print(name, pid)
        # print([ '%.2f' % elem for elem in memory])
        # print([ '%.2f' % elem for elem in memory_virtual])
        
        f=open("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + ".csv", "a")
        f.write(name+","+str(time.time())+","+str(np.mean(memory))+","+str(np.mean(memory_virtual))+","+filename+'\n')
        f.close()
        
def get_max_pid():
    processes = psutil.process_iter(['pid', 'name', 'memory_info'])
    # Initialize variables to track the highest memory usage
    max_memory = 0
    max_memory_pid = None
    
    # Iterate over the processes
    for process in processes:
        # Get the memory usage information for each process
        try:
            memory_info = process.info['memory_info']
            memory_usage = memory_info.rss
        except:
            memory_usage = 0
        # Check if the current process has higher memory usage
        if memory_usage > max_memory and 'python' in process.info['name'].lower():
            max_memory = memory_usage
            max_memory_name = process.info['name']
            max_memory_pid = process.info['pid']
    return max_memory_name, max_memory_pid
           
            
algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

MemTest(algo, mode, system)
# MemTest("DBSCAN", "SS", "M2")
