import os
import pandas as pd
import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import sys
from SS_Clustering import SS_Clustering
# from sklearn.decomposition import PCA
import psutil
import threading
import pdb
from sklearn.metrics.cluster import adjusted_rand_score
            


import openml
import openml.config
openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]
did = sys.argv[4]
filename = sys.argv[5]


def MTest_Run(algo, mode, system, did, filename):
    if system == "Jimmy":
        new_home_directory = '/jimmy/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    elif system == "Louise":
        new_home_directory = '/louise/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    elif system == "Thelma":
        new_home_directory = '/thelma/hdd/ma234/Temp/'
        openml.config.set_cache_directory(new_home_directory)
    
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
        writeTimeFile(filename, 0, 0, 0, 0, -1) # Other Errors or Invalid Dataset = -1
        return                    
    if all(is_numeric):                
        runFile(filename, df, algo, mode, system)
    else:
        print("In dataset ", filename, did, "non numaric columns exists (", sum(is_numeric), "out of", len(is_numeric), ")")
        writeTimeFile(filename, 0, 0, 0, 0, -1) # Other Errors or Invalid Dataset = -1
        
def runFile(file, df, algo, mode, system):
    r = df.shape[0]
    c = df.shape[1]
    gt_available = True
    if "target" in df.columns:
        y=df["target"].to_numpy()
        X=df.drop("target", axis=1)
        c=c-1
    elif "class" in df.columns:
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)
        c=c-1
    else:
        gt_available = False
        y = [0]*r
        X = df
    X.fillna(X.mean(numeric_only=True).round(1), inplace=True)
    if c < 10:
        print("#Column too low")
        writeTimeFile(file, 0, 0, 0, 0, -1) # Other Errors or Invalid Dataset = -1
        return

    print("Dataset size:", r,c)
    
    try:
        executed = 0
        
        t0 = time.time()
        if mode == "Default":
            
            
            
            writeTimeFile(filename, r, c, t0, 0, -22) # To see if it started or not = -22
            
            if algo == "DBSCAN":
                clustering = DBSCAN(algorithm="brute").fit(X)
                l = clustering.labels_
                outliers = np.count_nonzero((l==-1))
                uniq = len(set(l))
                f=open("Stats/" + algo + "/"+ system + "_Uniq&Outlier.csv", "a")
                f.write(file+','+str(uniq)+','+str(outliers)+'\n')
                f.close()
            elif algo == "AP":
                clustering = AffinityPropagation().fit(X)
                l = clustering.labels_
            elif algo == "GMM":
                clustering = GaussianMixture(n_components=2).fit(X)
                l = clustering.predict(X)
                
                
            elif algo == "HAC":
                clustering = AgglomerativeClustering().fit(X)
                l = clustering.labels_
            # df["predicted_labels"] = l
            # df.to_csv("../AcceleratedUL_Output/"+file+"_"+algo+"_"+mode+"_"+system+".csv")
            ari = adjusted_rand_score(y,l)
            time_ = time.time()-t0    
            
        else:
            clustering = SS_Clustering(algoName=algo)
            clustering.X = X
            clustering.y = y
            ari, time_ = clustering.run()
            clustering.destroy()
        
        if gt_available:           
            f=open("Stats/" + algo + "/"+ system + ".csv", "a")
            f.write(file+','+str(r)+','+str(c)+','+mode+','+system+','+str(time_)+','+str(ari)+'\n')
            f.close()
            
        writeTimeFile(file, r, c, t0, time.time(), 1) # Successful = 1
        
    except MemoryError:
        try:
            clustering.destroy()
        except:
            print()
        print(file, " killed due to low memory")
            
        writeTimeFile(file, r, c, t0, time.time(), 0) # Memory Error = 0

        
    except Exception as e:
        try:
            clustering.destroy()
        except:
            pass
        print(file + " killed. Reason: ", e)
        
        writeTimeFile(file, r, c, t0, time.time(), -1) # Other Errors or Invalid Dataset = -1
    



def writeTimeFile(filename, r, c, t0, t1, status):
    f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
    f.write(filename+','+str(r)+','+str(c)+','+str(t0)+','+str(t1)+','+str(status)+'\n')
    f.close()

if __name__ == '__main__':
    MTest_Run(algo, mode, system, did, filename)
    
    