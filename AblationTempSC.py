from PAU.PAU_Clustering import PAU_Clustering
import os
import sys
import glob
import pandas as pd
import time
import numpy as np

folderpath = '../Openml/'


        
        
    
def ReadFile(file):
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
    
    return X, y

def TestBatchSize(algo, X, y, filename):
    r = X.shape[0]
    c = X.shape[1]
    r_iteration = 3
    if algo == "AP":
        if r > 200000:
            return
        r_iteration = 1
        
    if algo == "SC": 
        if r > 200000:
            return
        r_iteration = 5
    for BatchSize in range(100,1501,100):
        BatchCount = int(r/BatchSize)
        # BatchSize = int(r/BatchCount)
        clustering = PAU_Clustering(algoName=algo, batch_count=BatchCount)
        clustering.X = X
        clustering.y = y
        aris=[]
        times=[]
        for i in range(r_iteration):
            ari, time_ = clustering.run()
            aris.append(ari)
            times.append(time_)
        clustering.destroy()
        ari = np.mean(aris)
        time_ = np.mean(times)
        
        f=open("Stats/Ablation/BatchSizeTest_" + algo + ".csv", "a")
        f.write(file+','+str(r)+','+str(c)+','+str(time_)+','+str(ari)+','+str(BatchCount)+','+str(BatchSize)+'\n')
        f.close()

def TestRefereeClAlgo(algo, X, y, filename):
    print(filename, end=" ")
    params_cl_algos = ["AP", "KM", "DBS", "HAC", "INERTIA"]
    for params_cl_algo in params_cl_algos:
        if params_cl_algo == algo:
            continue
        print(params_cl_algo)
        if algo == "AP":
            bc = int(X.shape[0]/100)
        else:
            bc = 0
        
        clustering = PAU_Clustering(algoName=algo, batch_count=bc)
        clustering.X = X
        clustering.y = y
        clustering.determine_param_clustering_algo = params_cl_algo
        
        aris=[]
        times=[]
        for i in range(1):
            ari, time_ = clustering.run()
            aris.append(ari)
            times.append(time_)
        clustering.destroy()
        ari = np.mean(aris)
        time_ = np.mean(times)
        
        f=open("Stats/Ablation/Ablation_RefereeClAlgo_" + algo + ".csv", "a")
        f.write(filename+','+str(params_cl_algo)+','+str(time_)+','+str(ari)+'\n')
        f.close()


def TestMode(algo, X, y, filename):
    print(filename, end=" ")
    modes = ["A", "B"]
    for mode in modes:
        print(mode)
        clustering = PAU_Clustering(algoName=algo)
        clustering.X = X
        clustering.y = y
        clustering.rerun_mode = mode
        
        aris=[]
        times=[]
        for i in range(5):
            ari, time_ = clustering.run()
            aris.append(ari)
            times.append(time_)
        clustering.destroy()
        ari = np.mean(aris)
        time_ = np.mean(times)
        
        f=open("Stats/Ablation/Ablation_Mode_" + algo + ".csv", "a")
        f.write(filename+','+str(mode)+','+str(time_)+','+str(ari)+'\n')
        f.close()
    

def InitStatsFile(algo, test):
    done_files = []
    if os.path.isdir("Stats/Ablation/") == 0:
        os.mkdir("Stats/Ablation/")
        
    if test == "Batch":
        if os.path.exists("Stats/Ablation/BatchSizeTest_" + algo + ".csv") == 0:
            f=open("Stats/Ablation/BatchSizeTest_" + algo + ".csv", "w")
            f.write('Filename,Row,Column,Time,ARI,BatchCount,BatchSize\n')
            f.close()
        else:
            done_files = pd.read_csv("Stats/Ablation/BatchSizeTest_" + algo + ".csv")
            done_files = done_files["Filename"].to_numpy()
            return done_files
        
    elif test == "Referee":
        if os.path.exists("Stats/Ablation/Ablation_RefereeClAlgo_" + algo + ".csv") == 0:
            f=open("Stats/Ablation/Ablation_RefereeClAlgo_" + algo + ".csv", "w")
            f.write('Filename,Referee,Time,ARI\n')
            f.close()
        else:
            done_files = pd.read_csv("Stats/Ablation/Ablation_RefereeClAlgo_" + algo + ".csv")
            done_files = done_files["Filename"].to_numpy()
            return done_files
    elif test == "Mode":
        if os.path.exists("Stats/Ablation/Ablation_Mode_" + algo + ".csv") == 0:
            f=open("Stats/Ablation/Ablation_Mode_" + algo + ".csv", "w")
            f.write('Filename,Mode,Time,ARI\n')
            f.close()
        else:
            done_files = pd.read_csv("Stats/Ablation/Ablation_Mode_" + algo + ".csv")
            done_files = done_files["Filename"].to_numpy()
            return done_files
            
    return None
    
if __name__ == '__main__':
    algo = sys.argv[1]
    test = sys.argv[2]
    # test = "Referee"
    
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1]
        master_files[i] = master_files[i][:-4]
    
    done_files = InitStatsFile(algo, test)
    datasets_of_interest = pd.read_csv("Stats/Merged_SS.csv")["Filename"].to_numpy()
    if done_files is not None:
        master_files = [x for x in master_files if x not in done_files]
    master_files = [x for x in master_files if x in datasets_of_interest] 
    
    master_files.sort()
    
    fastfiles = ["BNG(2dplanes)_OpenML",
                 "Diabetes130US_OpenML", 
                 "spoken-arabic-digit_OpenML", 
                 "BNG(pwLinear)_OpenML"]
    
    for file in master_files:
        if file not in fastfiles:
            continue
        X, y = ReadFile(file)
        
        if test == "Referee":
            TestRefereeClAlgo(algo, X, y, file)
        elif test == "Batch":
            TestBatchSize(algo, X, y, file)
        elif test == "Mode":
            TestMode(algo, X, y, file)
    











