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

        
def mergeWithARI():
    algo_list = pd.read_csv("Stats/DBSCAN_Algo_Choice.csv")
    fileList = pd.read_csv("../MemoryStats/FileList.csv")
    aris = pd.read_csv("../Stats/Merged_Default_Filtered.csv")
    aris = aris.loc[:, ['Filename', 'ARI_DBSCAN']]
    
    algo_list = algo_list[algo_list['Filename'].isin(fileList['Filename'])]

    merged_df = pd.merge(algo_list, aris, on='Filename', how='outer')
    merged_df.to_csv("Stats/DBSCAN_Algo_Choice.csv",index=False)
    # print(merged_df)
    
mergeWithARI()

# getAlgo()