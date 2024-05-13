import os
import pandas as pd
import numpy as np
import sys
import glob

algo = sys.argv[1]
system = sys.argv[2]
mode = "ACE"

if __name__ == '__main__':
    folderpath = '../Openml/'
    done_files = []
    if os.path.exists("Stats/" + algo + "/"+ system + ".csv") == 0:
        if os.path.isdir("Stats/" + algo + "/") == 0:    
            os.mkdir("Stats/" + algo + "/")
        f=open("Stats/" + algo + "/"+ system + ".csv", "w")
        f.write('Filename,Row,Columm,Mode,System,Time,ARI\n')
        f.close()
    else:
        done_files = pd.read_csv("Stats/" + algo + "/"+ system + ".csv")
        done_files = done_files["Filename"].to_numpy()

    if os.path.exists("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv"):
        df_done_files = pd.read_csv("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv")
        df_done_files = df_done_files["Filename"].to_numpy()
        done_files = np.concatenate((done_files, df_done_files), axis=0)
    else:
        f=open("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv", "w")
        f.write('Filename,Row,Columm,StartTime,EndTime,Completed\n')
        f.close()
    
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1]
        master_files[i] = master_files[i][:-4]
    master_files = [x for x in master_files if x not in done_files] 
    master_files.sort()

    fileList = pd.read_csv("MemoryStats/FileList.csv")
    fileList = fileList["Filename"].to_numpy()
    
    master_files = [value for value in master_files if value in fileList]
    

    """Only run R_algo"""
    df_Default = pd.read_csv("Stats/Merged_Default_Filtered.csv")
    D_file_list = []

    for index, row in df_Default.iterrows():
        if pd.notna(row["ARI_"+algo]):
            D_file_list.append(row['Filename'])
    master_files = [value for value in master_files if value in D_file_list]

    # print(master_files)

    with open('ace_master_files.txt', 'w') as f:
        for item in master_files:
            f.write(f"{item}\n")