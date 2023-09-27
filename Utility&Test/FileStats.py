import os
import glob
import pandas as pd

# from memory_profiler import profile
import warnings 
warnings.filterwarnings("ignore")

from pathlib import Path

datasetFolderDir = '../Openml/'


def readData(filename):
    df = pd.read_csv(datasetFolderDir+filename+".csv")
    row = df.shape[0]
    col = df.shape[1]
    
    if "target" in df.columns:
        y=df["target"].to_numpy()
        X=df.drop("target", axis=1)
    elif "class" in df.columns:
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)

    u = len(set(y))
    
    f=open("Stats/FileStats_OpenML.csv", "a")
    f.write(filename+","+str(row)+","+str(col)+","+str(u)+"\n")
    f.close()

    
if __name__ == '__main__':
    algorithm = "DBSCAN"
    
    folderpath = datasetFolderDir
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        # master_files[i] = master_files[i].split("/")[-1].split(".")[0]
        master_files[i] = Path(master_files[i]).stem
    master_files.sort()
    
    print(master_files)
    filelist = pd.read_csv("MemoryStats/FileList.csv")
    
    
    
    
    # if os.path.exists("Stats/FileStats_OpenML.csv") == 0:
    #     f=open("Stats/FileStats_OpenML.csv", "w")
    #     f.write('Filename,Shape_R,Shape_C,Unique\n')
    #     f.close()
    
    # remaining = len(master_files)
    # for file in master_files:
        
    #     readData(file)
        
    #     print(remaining)
    #     remaining-=1