import glob
import pandas as pd
from sklearn.cluster import DBSCAN

def getAlgo():
    
    folderpath = '../../Openml/'
    # folderpath = '/Users/muyeedahmed/Desktop/Gitcode/Openml/'
    master_files = glob.glob(folderpath+"*.csv")

    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1]
        master_files[i] = master_files[i][:-4]
    # master_files = [x for x in master_files if x not in done_files] 
    master_files.sort()
    
    for fn in master_files:
        df = pd.read_csv(folderpath+fn+".csv")
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)
        print(fn,end='')
        try:            
            c = DBSCAN().fit(X)
        except:
            pass

        
getAlgo()