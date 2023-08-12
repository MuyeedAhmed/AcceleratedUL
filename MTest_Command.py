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

from collections import OrderedDict
openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


def MemTest(algo, mode, system):
    if system == "M2":
        folderpath = '../Openml/'
    elif system == "Jimmy":
        folderpath = '/jimmy/hdd/ma234/Openml/'
    elif system == "Louise":
        folderpath = '/louise/hdd/ma234/Openml/'
    elif system == "3070":
        folderpath = '../Openml/'
    elif system == "2060":
        folderpath = '../Openml/'
    elif system == "Thelma":
        folderpath = "/thelma/hdd/ma234/Openml/"
    else:
        print("System name doesn't exist")
        return
    
    
    
    done_files = []
    if os.path.exists("Stats/" + algo + "/"+ system + ".csv") == 0:
        if os.path.isdir("Stats/" + algo + "/") == 0:    
            os.mkdir("Stats/" + algo + "/")
        f=open("Stats/" + algo + "/"+ system + ".csv", "w")
        f.write('Filename,Row,Columm,Mode,System,Time,ARI\n')
        f.close()
    else:
        done_files = pd.read_csv("Stats/" + algo + "/"+ system + ".csv")
        done_files = done_files["Filename"].to_numpy()

    # done_files = []
    if os.path.exists("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv"):
        df_done_files = pd.read_csv("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv")
        df_done_files = df_done_files["Filename"].to_numpy()
        done_files = np.concatenate((done_files, df_done_files), axis=0)
    else:
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "w")
        f.write('Filename,Row,Columm,StartTime,EndTime,Completed\n')
        f.close()
    
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1]
        master_files[i] = master_files[i][:-4]
    master_files = [x for x in master_files if x not in done_files] 
    master_files.sort()
    
    ## For AP and SC Default
    algoTime = pd.read_csv("Stats/Time/"+algo+"/"+system+".csv")
    
    for filename in master_files:
        print(filename)
        if filename in algoTime["Filename"].values:
            est = algoTime[algoTime["Filename"]==filename]["Estimated_Time"].to_numpy()[0]
            if est > 3000:
                continue
        """
        Kill previous process
        
        while True:
            p_name, p_id, mem = get_max_pid()
            if mem > 100000:
                command = "kill -9 " + str(p_id)
                os.system(command)
            else:
                break
        """
        
        argument = [algo, mode, system, filename]
        command = ["python", "MTest_RunData.py"] + argument

        
        stop_flag = threading.Event()
        MonitorMemory = threading.Thread(target=monitor_memory_usage_pid, args=(algo, mode, system, filename,stop_flag,))
        MonitorMemory.start()
        
        try:
            subprocess.run(command, timeout=7200)
        except subprocess.TimeoutExpired:
            f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
            f.write(filename+',0,0,0,0,-23\n')
            f.close()
            print("Timed out")
        
        stop_flag.set()
        MonitorMemory.join()
        
        done_files += filename

        command = "import gc; gc.collect()"
        subprocess.run(["python", "-c", command])
                

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
        name, pid, _ = get_max_pid()
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
            memory_usage = memory_info.vms / (1024 * 1024)
        except:
            memory_usage = 0
        # Check if the current process has higher memory usage
        if memory_usage > max_memory and 'python' in process.info['name'].lower():
            max_memory = memory_usage
            max_memory_name = process.info['name']
            max_memory_pid = process.info['pid']
    return max_memory_name, max_memory_pid, max_memory
           
               
algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

MemTest(algo, mode, system)
# MemTest("AP", "Default", "M2")
