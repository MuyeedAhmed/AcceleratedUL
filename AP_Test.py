import os
import pandas as pd
import numpy as np
import random
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import sys
from SS_Clustering import SS_Clustering

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
import matplotlib.pyplot as plt

from collections import OrderedDict
openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


def MemoryTest_List(algo, mode, system):
    
        
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
        
        t0 = time.time()
        p = multiprocessing.Process(target=worker, args=(d, algo, mode, system, row, c, file,))
        p.start()
        p.join(timeout=7200)
        
        if p.is_alive():
            p.terminate()
            print("Terminated")
            t1 = time.time()
            executed = -1
            f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + "_Row.csv", "a")
            f.write(file+','+str(row)+','+str(c)+','+str(t0)+','+str(t1)+','+str(executed)+'\n')
            f.close()
        
        
        
        command = "import gc; gc.collect()"
        subprocess.run(["python", "-c", command])
        
        time.sleep(5)


def getThreshold(algo, mode, system):

    file = "bot-iot-all-features"
    # file = "KDDCup99"
    # file = "delays_zurich_transport"
    dataset = openml.datasets.get_dataset(file)
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
        )
    df = pd.DataFrame(X)
    # df["class"] = y
    c = df.shape[1]
    print(df.shape)
    start = 260000
    
    end = 290000
    while True:
        while True:
            p_name, p_id, mem = get_max_pid()
            print(p_name, p_id, mem )
            if mem > 100000:
                command = "kill -9 " + str(p_id)
                print(command)
                os.system(command)
            else:
                break
        done = multiprocessing.Value('i', 0)
        if start >= end:
            print("Threshold: ", start)
            break
        mid =int((end+start)/2)+1
        d = df.iloc[:mid].copy()
        print("mid: ", mid)
        t0 = time.time()
        p = multiprocessing.Process(target=worker, args=(d, algo, mode, system, mid, c, file, done,))
        p.start()
        p.join(timeout=20000)
        if p.is_alive():
            p.terminate()
            print("Terminated: ", mid)
            start = mid
            time.sleep(30)
        else:
            if done.value == 1:
                print("Done in time: ", time.time() - t0)
                start = mid            
            else:
                print("Killed in time: ", time.time() - t0)
                end = mid
                time.sleep(30)
        
        
        command = "import gc; gc.collect()"
        subprocess.run(["python", "-c", command])
        
        time.sleep(10)
        
        
        
def worker(d, algo, mode, system, row, c, file, done):
    t0 = time.time()
    done.value = 0
    try:
        executed = 0
        if mode == "Default":
            if algo == "DBSCAN":
                clustering = DBSCAN(algorithm="brute").fit(d)
            elif algo == "AP":
                clustering = AffinityPropagation().fit(d)
            elif algo == "GMM":
                clustering = GaussianMixture(n_components=2).fit(d)
            elif algo == "SC":
                clustering = SpectralClustering(n_clusters=2).fit(d)
            elif algo == "HAC":
                clustering = AgglomerativeClustering().fit(d)
                # print(clustering.labels_)
        else:
            clustering = SS_Clustering(algoName=algo)
            clustering.X = d
            clustering.y = [0]*d.shape[0]
            clustering.run()
            clustering.destroy()
        done.value = 1
        t1 = time.time()
        
        executed = 1
        # f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + "_Row.csv", "a")
        # f.write(file+','+str(row)+','+str(c)+','+str(t0)+','+str(t1)+','+str(executed)+'\n')
        # f.close()
        
        
    except MemoryError:
        t1 = time.time()
        # f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + "_Row.csv", "a")
        # f.write(file+','+str(row)+','+str(c)+','+str(t0)+','+str(t1)+','+str(executed)+'\n')
        # f.close()
        print(file, " killed due to low memory")

def monitor_memory_usage_pid(algo, mode, system, filename, stop_flag):
    print("Memory usage")
    interval = 0.1
    if os.path.exists("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + "_Row.csv") == 0:
        f=open("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + "_Row.csv", "w")
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
        
        f=open("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + "_Row.csv", "a")
        f.write(name+","+str(time.time())+","+str(np.mean(memory))+","+str(np.mean(memory_virtual))+","+filename+'\n')
        f.close()
        
def get_max_pid():
    processes = psutil.process_iter(['pid', 'name', 'memory_info'])
    max_memory = 0
    max_memory_pid = None
    
    for process in processes:
        try:
            memory_info = process.info['memory_info']
            memory_usage = memory_info.vms / (1024 * 1024)
        except:
            memory_usage = 0
        if memory_usage > max_memory and 'python' in process.info['name'].lower():
            max_memory = memory_usage
            max_memory_name = process.info['name']
            max_memory_pid = process.info['pid']
    return max_memory_name, max_memory_pid, max_memory
           



def MemoryConsumptionCalculation(algo, mode, system):
    memory = pd.read_csv("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + "_Row.csv")
    time = pd.read_csv("MemoryStats/Time_" + algo + "_" + mode + "_" + system + "_Row.csv") 
    
    time["TotalTime"] = time["EndTime"] - time["StartTime"]
    
    # time["Memory_Median"] = None
    time["Memory_Physical_Median"] = None
    time["Memory_Virtual_Median"] = None
    
    # time["Memory_Max"] = None
    time["Memory_Physical_Max"] = None
    time["Memory_Virtual_Max"] = None
    
    for index, row in time.iterrows():
        t = memory[(memory["Time"] > row["StartTime"]) & (memory["Time"] < row["EndTime"])]
            # print(t)
        if t.empty:
            print(row["Filename"])
            continue
        
        memory_physical = t["Memory_Physical"].to_numpy()
        mp_median = np.median(memory_physical)
        mp_max = np.max(memory_physical)
        
        memory_virtual = t["Memory_Virtual"].to_numpy()
        mv_median = np.median(memory_virtual)
        mv_max = np.max(memory_virtual)
        
        # time.loc[index, "Memory_Median"] = mp_median + mv_median
        time.loc[index, "Memory_Physical_Median"] = mp_median
        time.loc[index, "Memory_Virtual_Median"] = mv_median
        
        # time.loc[index, "Memory_Max"] = mp_max + mv_max
        time.loc[index, "Memory_Physical_Max"] = mp_max
        time.loc[index, "Memory_Virtual_Max"] = int(mv_max)
    
    time.to_csv("MemoryStats/Time_Memory_" + algo + "_" + mode + "_" + system + "_Row.csv")
    
def drawGraph(algo, system):
    default = pd.read_csv("MemoryStats/Time_Memory_" + algo + "_Default_" + system + "_Row.csv")
    ss = pd.read_csv("MemoryStats/Time_Memory_" + algo + "_SS_" + system + "_Row.csv")

    draw(default, ss, "Memory_Virtual_Max", algo, system)
    draw(default, ss, "TotalTime", algo, system)
    
def draw(df_d, df_s, tm, algo, system):    
    df_s = df_s[df_s['Filename'].isin(df_d['Filename'])]
    df_d = df_d[df_d['Filename'].isin(df_s['Filename'])]
    
    x_Default = df_d["Row"]
    x_SS = df_s["Row"]
    
    y_Default = df_d[tm]
    y_SS = df_s[tm]
    
    plt.figure(0)
    plt.plot(x_SS,y_SS, ".",color="blue")
    plt.plot(x_Default,y_Default, ".",color="red")
    
    # AP
    # row = math.sqrt((memory_size * 10**9)/ (7 * 4)) ## AP each value 4 bytes
    plt.axvline(x = 169000, color='red', linestyle = '-') # Jimmy 800
    plt.axvline(x = 110000, color='orange',linestyle = '--') # Thelma 340
    plt.axvline(x = 84000, color='purple',linestyle = '--') # M2 200
    plt.axvline(x = 80000, color='cyan', linestyle = '--') # Louise 180
    
    # SC
    # row = math.sqrt((memory_size * 10**9)/ (4 * 8)) ## SC each value 8 bytes
    # plt.axvline(x = 158000, color='red', linestyle = '-') # Jimmy 800
    # plt.axvline(x = 103000, color='orange',linestyle = '--') # Thelma 340
    # plt.axvline(x = 79000, color='purple',linestyle = '--') # M2 200
    # plt.axvline(x = 75000, color='cyan', linestyle = '--') # Louise 180


    # HAC
    # plt.axvline(x = , color='red', linestyle = '-') # Jimmy 800
    # plt.axvline(x = 223442, color='orange',linestyle = '--') # Thelma 340
    # plt.axvline(x = 112834, color='purple',linestyle = '--') # M2 200
    # plt.axvline(x = , color='cyan', linestyle = '--') # Louise 180
    
    
    plt.grid(True)
    plt.legend(["Subsampling", "Default", "Jimmy", "Thelma", "M2", "Louise"])
    plt.xlabel("Points (Rows)")
    # plt.xlim([0, 500000])
    if tm == "Memory_Virtual_Max":
        plt.ylabel("Memory (in MB)")
        plt.title(algo + " Memory Usage in " + system)
    else:
        plt.ylabel("Time (in Seconds)")
        plt.title(algo + " Execution Time in " + system)
    
    plt.savefig('MemoryStats/Figures/'+tm+'_' + algo + '_' + system +'_Row.pdf', bbox_inches='tight')
    plt.show()

    


               
algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

if __name__ == '__main__':
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
        
    # MemoryTest_List(algo, mode, system)
    
    getThreshold(algo, mode, system)

    
    # algo = "AP"
    # system = "Jimmy"

    # mode = "SS"
    # MemoryConsumptionCalculation(algo, mode, system)
    # mode = "Default"
    # MemoryConsumptionCalculation(algo, mode, system)

    # drawGraph(algo, system)

    

