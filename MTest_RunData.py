import os
import pandas as pd
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import sys
from SS_Clustering import SS_Clustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import psutil
import threading
import pdb
            


import openml
import openml.config
openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]
did = sys.argv[4]
filename = sys.argv[5]


def MTest_Run(algo, mode, system, did, filename):
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
    # else:
    #     print("System name doesn't exist")
    #     return
    
    try:
        dataset = openml.datasets.get_dataset(did)
        
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
            )
        df = pd.DataFrame(X)
        df["class"] = y
        is_numeric = df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    except:
        print("Failed to read data: ", did)
        writeFailed(filename)
        return                    
    if all(is_numeric):                
        # stop_flag = threading.Event()
        # MonitorMemory = threading.Thread(target=monitor_memory_usage_pid, args=(algo, mode, system, filename,stop_flag,))
        
        # MonitorMemory.start()
        
        runFile(filename, df, algo, mode, system)

        # stop_flag.set()
        
        # MonitorMemory.join()
        
        
    else:
        print("In dataset ", filename, did, "non numaric columns exists (", sum(is_numeric), "out of", len(is_numeric), ")")
        

def runFile(file, df, algo, mode, system):
    r = df.shape[0]
    c = df.shape[1]

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
    if c < 10:
        print("#Column too low")
        writeFailed(filename)
        return

    print("Dataset size:", r,c)
    
    try:
        executed = 0
        
        t0 = time.time()
        if mode == "Default":
            
            """
            To see if it started or not
            """
            print("Start default run")
            f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
            f.write(file+','+str(r)+','+str(c)+','+str(t0)+',0,-22\n')
            f.close()
            
            if algo == "DBSCAN":
                clustering = DBSCAN(algorithm="brute").fit(X)
                l = clustering.labels_
            elif algo == "AP":
                clustering = AffinityPropagation().fit(X)
                l = clustering.labels_
            elif algo == "GMM":
                clustering = GaussianMixture(n_components=2).fit(X)
                l = clustering.predict(X)
            df["predicted_labels"] = l
            df.to_csv("ClusteringOutput/"+file+"_"+algo+"_"+mode+"_"+system+".csv")
            
            
            print("*Done*")
        else:
            clustering = SS_Clustering(algoName=algo)
            clustering.X = X
            clustering.y = y
            clustering.run()
            clustering.destroy()
        t1 = time.time()
        executed = 1
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
        f.write(file+','+str(r)+','+str(c)+','+str(t0)+','+str(t1)+','+str(executed)+'\n')
        f.close()
    except MemoryError:
        try:
            clustering.destroy()
        except:
            print()
        t1 = time.time()
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
        f.write(file+','+str(r)+','+str(c)+','+str(t0)+','+str(t1)+','+str(executed)+'\n')
        f.close()
        print(file, " killed due to low memory")
    except Exception as e:
        try:
            clustering.destroy()
        except:
            print()
        print(file + " killed. Reason: ", e)
        executed = -1
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
        f.write(file+','+str(r)+','+str(c)+','+str(t0)+','+str(time.time())+','+str(executed)+'\n')
        f.close()
    
def runDefault(algo, X):
    if algo == "DBSCAN":
        clustering = DBSCAN(algorithm="brute").fit(X)
    elif algo == "AP":
        clustering = AffinityPropagation().fit(X)
    elif algo == "GMM":
        clustering = GaussianMixture(n_components=2).fit(X)

def writeFailed(filename):
    f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
    f.write(filename+',0,0,0,0,-1\n')
    f.close()

if __name__ == '__main__':
    MTest_Run(algo, mode, system, did, filename)
    
    
    