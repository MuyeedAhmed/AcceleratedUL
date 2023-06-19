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
# import shuffle
import subprocess
import openml
import openml.config

from collections import OrderedDict
openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


def MemTest(algo, mode, system):
    # if system == "M2":
    #     folderpath = '/Users/muyeedahmed/Desktop/Research/Dataset/'
    if system == "Jimmy":
        # folderpath = '/jimmy/hdd/ma234/Dataset/'
        new_home_directory = '/jimmy/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    elif system == "Louise":
        # folderpath = '/louise/hdd/ma234/Dataset/'
        new_home_directory = '/louise/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    # elif system == "3070":
    #     folderpath = '../Datasets/'
    elif system == "Thelma":
        # folderpath = ""
        new_home_directory = '/thelma/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
        
    f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + "_Row.csv", "w")
    f.write('Filename,Row,Columm,StartTime,EndTime,Completed\n')
    f.close()
    
    file = "bot-iot-all-features"
    
    dataset = openml.datasets.get_dataset(file)
                    
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
        )
    df = pd.DataFrame(X)
    # df["class"] = y
    c = df.shape[1]
    print(df.shape)
    rows = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000, 3000000]
    rows.reverse()
    for row in rows:
        d = df.iloc[:row].copy()
        
        # stop_flag = threading.Event()
        # MonitorMemory = threading.Thread(target=monitor_memory_usage_pid, args=(algo, mode, system, str(row), stop_flag,))
        # MonitorMemory.start()
        
        t0 = time.time()
        p = multiprocessing.Process(target=worker, args=(d, algo, mode, system, row, c, file,))
        p.start()
        p.join(timeout=1800)
        
        if p.is_alive():
            p.terminate()
            print("Terminated")
            t1 = time.time()
            executed = -1
            f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + "_Row.csv", "a")
            f.write(file+','+str(row)+','+str(c)+','+str(t0)+','+str(t1)+','+str(executed)+'\n')
            f.close()
        
        
        # print("hue: ", row, t1-t0)
        

        
        # stop_flag.set()
        # MonitorMemory.join()
        
        command = "import gc; gc.collect()"
        subprocess.run(["python", "-c", command])
        
        time.sleep(5)
        
def worker(d, algo, mode, system, row, c, file):
    t0 = time.time()
    try:
        executed = 0
        if mode == "Default":
            # else:
            if algo == "DBSCAN":
                clustering = DBSCAN(algorithm="brute").fit(d)
            elif algo == "AP":
                clustering = AffinityPropagation().fit(d)
            elif algo == "GMM":
                clustering = GaussianMixture(n_components=2).fit(d)
        else:
            clustering = SS_Clustering(algoName=algo)
            clustering.X = d
            clustering.y = [0]*d.shape[0]
            clustering.run()
            clustering.destroy()
        t1 = time.time()
        executed = 1
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + "_Row.csv", "a")
        f.write(file+','+str(row)+','+str(c)+','+str(t0)+','+str(t1)+','+str(executed)+'\n')
        f.close()
        
    except MemoryError:
        t1 = time.time()
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + "_Row.csv", "a")
        f.write(file+','+str(row)+','+str(c)+','+str(t0)+','+str(t1)+','+str(executed)+'\n')
        f.close()
        print(file, " killed due to low memory")

def monitor_memory_usage_pid(algo, mode, system, filename, stop_flag):
    print("Memory usage")
    interval = 0.1
    # algo = sys.argv[1]
    # mode = sys.argv[2]
    # system = sys.argv[3]
    # filename = sys.argv[4]
    
    if os.path.exists("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + "_Row.csv") == 0:
        f=open("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + "_Row.csv", "w")
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
        
        f=open("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + "_Row.csv", "a")
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
# MemTest("DBSCAN", "Default", "M1")



