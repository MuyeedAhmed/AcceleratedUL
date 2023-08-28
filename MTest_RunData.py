import os
import pandas as pd
import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

import sys
from SS_Clustering import SS_Clustering
# from sklearn.decomposition import PCA
import psutil
import threading
import pdb
from sklearn.metrics.cluster import adjusted_rand_score


algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]
filename = sys.argv[4]


def MTest_Run(algo, mode, system, filename):
    instances_to = 2000000000
    if system == "M2":
        if algo == "AP":
            instances_to = 84000
        if algo == "SC":
            instances_to = 79000
        elif algo == "HAC":
            instances_to = 120000 ###
            
        folderpath = '../Openml/'
        
    elif system == "Jimmy":
        if algo == "AP":
            instances_to = 170000
        if algo == "SC":
            instances_to = 158000
        elif algo == "HAC":
            instances_to = 315000
            
        folderpath = '/jimmy/hdd/ma234/Openml/'
        
    elif system == "Louise":
        if algo == "AP":
            instances_to = 80000
        if algo == "SC":
            instances_to = 75000
        elif algo == "HAC":
            instances_to = 157000
            
        folderpath = '/louise/hdd/ma234/Openml/'
        
    elif system == "3070":
        folderpath = '../Openml/'
    elif system == "2060":
        folderpath = '../Openml/'
    elif system == "Thelma":
        if algo == "AP":
            instances_to = 110000
        if algo == "SC":
            instances_to = 103000
        elif algo == "HAC":
            instances_to = 220000
        
        folderpath = "/thelma/hdd/ma234/Openml/"
    else:
        print("System name doesn't exist")
        return
    

    df = pd.read_csv(folderpath+filename+".csv")
    if df.shape[0] > instances_to and mode == "Default":
        writeTimeFile(filename, 0, 0, 0, 0, -10) # Memory No available
        print("Too large for this algorithm")
    else:
        runFile(filename, df, algo, mode, system)
        
        
    
def runFile(file, df, algo, mode, system):
    r = df.shape[0]
    c = df.shape[1]
    gt_available = True
    
    if "class" in df.columns:
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)
        c=c-1
    else:
        print("Ground truth not available")
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
            elif algo == "SC":
                clustering = SpectralClustering().fit(X)
                l = clustering.labels_
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
    MTest_Run(algo, mode, system, filename)
    
    