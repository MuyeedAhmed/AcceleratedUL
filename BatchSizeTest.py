from PAU.PAU_Clustering import PAU_Clustering
import os
import sys
import glob
import pandas as pd
import time


folderpath = '../Openml/'


        
        
    
def runFile(file, df, algo):
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
        
    for BatchSize in range(100,1501,100):
        BatchCount = int(r/BatchSize)
        # BatchSize = int(r/BatchCount)
        clustering = PAU_Clustering(algoName=algo, batch_count=BatchCount)
        clustering.X = X
        clustering.y = y
        ari, time_ = clustering.run()
        clustering.destroy()
        
    
        f=open("Utility&Test/Stats/BatchSizeTest_" + algo + ".csv", "a")
        f.write(file+','+str(r)+','+str(c)+','+str(time_)+','+str(ari)+','+str(BatchCount)+','+str(BatchSize)+'\n')
        f.close()

if __name__ == '__main__':
    master_files = glob.glob(folderpath+"*.csv")
    algo = "HAC"
    
    if os.path.exists("Utility&Test/Stats/BatchSizeTest_" + algo + ".csv") == 0:
        f=open("Utility&Test/Stats/BatchSizeTest_" + algo + ".csv", "a")
        f.write('Filename,Row,Column,Time,ARI,BatchCount,BatchSize\n')
        f.close()

    for file in master_files:
        if "solar-flare" not in file:
            continue
        df = pd.read_csv(file)
        
        runFile(file, df, algo)
    
    

