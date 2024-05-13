import os
import pandas as pd
import numpy as np
import time

import sys
from PAU.PAU_Clustering import PAU_Clustering
import psutil
import threading
import pdb
from sklearn.metrics.cluster import adjusted_rand_score


algo = sys.argv[1]
system = sys.argv[2]
filename = sys.argv[3]

mode="ACE"
    
def _run_ACE(algo, system, file):
    folderpath = '../Openml/'
    df = pd.read_csv(folderpath+file+".csv")
    
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

    # print("Dataset size:", r,c)
    t0 = time.time()
    try:
        executed = 0
        
        clustering = PAU_Clustering(algoName=algo, fileName=file)
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
        # print(file, " killed due to low memory")            
        writeTimeFile(file, r, c, t0, time.time(), 0) # Memory Error = 0
    except Exception as e:
        try:
            clustering.destroy()
        except:
            pass
        # print(file + " killed. Reason: ", e)
        
        writeTimeFile(file, r, c, t0, time.time(), e) # Other Errors or Invalid Dataset = -1
    
def writeTimeFile(filename, r, c, t0, t1, status):
    f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "a")
    f.write(filename+','+str(r)+','+str(c)+','+str(t0)+','+str(t1)+','+str(status)+'\n')
    f.close()

if __name__ == '__main__':    
    # _run_ACE(algo, system, filename)
    print("",end='')
    
    