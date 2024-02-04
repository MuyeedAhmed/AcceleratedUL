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
    algo_list = algo_list[algo_list['Filename'].isin(fileList['Filename'])]
    
    aris_def = pd.read_csv("../Stats/Merged_SS.csv")
    aris_def = aris_def.loc[:, ['Filename', "Row", "Columm",'ARI_DBSCAN','Time_DBSCAN']]
    
    algo_ari = pd.merge(algo_list, aris_def, on='Filename', how='outer')

    
    time_def = pd.read_csv("../MemoryStats/DBSCAN/Time_Memory_DBSCAN_Default_Jimmy_All.csv")
    time_ss =  pd.read_csv("../MemoryStats/DBSCAN/Time_Memory_DBSCAN_SS_Jimmy_All.csv")
    time_def = time_def.loc[:, ['Filename', 'TotalTime']]
    time_ss = time_ss.loc[:, ['Filename', 'TotalTime']]
    time_ = pd.merge(time_def, time_ss, on='Filename', how='outer',suffixes=('_Default', '_SAC'))
    
    # print(time_)
    
    
    algo_ari_time = pd.merge(algo_ari, time_, on='Filename', how='outer')
    
    algo_ari_time.to_csv("Stats/DBSCAN_Algo_Choice_With_ARI.csv",index=False)
    # print(merged_df)
    
mergeWithARI()

# getAlgo()