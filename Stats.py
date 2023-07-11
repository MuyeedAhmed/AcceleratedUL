import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt 
from scipy.stats import ttest_ind






def plot_time(df_default, df_ss):
    diff = []
    x = []
    y = []
    t = []
    for index, row in df_default.iterrows():
        ss_ari = df_ss[df_ss["Filename"] == row["Filename"]]
        if ss_ari.empty:
            print(row["Filename"], " not there in SS")
            continue
        x.append(row["Row"])
        y.append(row["Columm"])
        t.append(row["Time"])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.view_init(elev=60, azim=270, roll=0)
    ax.scatter(x, y, t)
        
def calculate_ari_diff(df_default, df_ss):
    diff = []
    row = []
    for index, row in df_default.iterrows():
        ss_ari = df_ss[df_ss["Filename"] == row["Filename"]]
        if ss_ari.empty:
            print(row["Filename"], " not there in SS")
            continue
        diff.append(ss_ari["ARI"].to_numpy()[0] - row['ARI'])
        row.append(row["Row"])
    #plot with row
    plt.figure()
    plt.plot(row, diff, ".")
    #plot sorted
    plt.figure()
    diff.sort()
    plt.plot(diff)
    
## P-value
def ttest(df_default, df_ss):
    default = []
    ss = []
    for index, row in df_default.iterrows():
        ss_ari = df_ss[df_ss["Filename"] == row["Filename"]]
        if ss_ari.empty:
            print(row["Filename"], " not there in SS")
            continue
        default.append(row["ARI"])
        ss.append(ss_ari["ARI"].to_numpy()[0])
        
        
    print(len(ss), len(default))
    t_statistic, p_value = ttest_ind(ss, default)
    
    print(t_statistic, p_value)


df_default = pd.read_csv("Stats/HAC/Louise.csv")
df_ss = pd.read_csv("Stats/HAC/Thelma.csv")

# plot_time(df_default, df_ss)
# calculate_ari_diff(df_default, df_ss)
ttest(df_default, df_ss)



def MemoryConsumptionCalculation(algo, mode, system):

    memory = pd.read_csv("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + ".csv")
    time = pd.read_csv("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv") 
    
    
    
    
    time = time[time["Completed"] == 1]
    
    time["TotalTime"] = time["EndTime"] - time["StartTime"]
    time['TotalTime'] = time['TotalTime'].mask(time['TotalTime'] < 0, np.nan)

    time["Memory_Max"] = None
    
    
    for index, row in time.iterrows():
        t = memory[(memory["Time"] > row["StartTime"]) & (memory["Time"] < row["EndTime"])]
        if t.empty:
            print(row["Filename"], " is empty!")
            continue
        
        memory_virtual = t["Memory_Virtual"].to_numpy()
        mv_max = np.max(memory_virtual)
        
        time.loc[index, "Memory_Max"] = int(mv_max)
    # print(time)
    
    """"""
    lrd = pd.read_csv("Stats/DBSCAN/M2_lrd.csv")
    label_stats = pd.read_csv("Stats/DBSCAN/M2_Uniq&Outlier.csv") 
    """"""
    
    
    dbscan = time.join(label_stats.set_index('Filename'), lsuffix='_caller', rsuffix='_other', on='Filename')
    
    print(lrd)
    print(dbscan)
    
    dbscan = dbscan.set_index('Filename').join(lrd.set_index('Filename'), lsuffix='_caller', rsuffix='_other')
    
    # dbscan = pd.concat([time, label_stats], axis=1, join="inner", ignore_index=True)
    
    
    dbscan.to_csv("Stats/DBSCAN/With Memory.csv")
    
    # table = time.pivot(index='Row', columns='Columm', values='Memory_Max')
    # table.to_csv("Max_Memory_Usage_" + algo + "_" + mode + "_" + system + ".csv")

# MemoryConsumptionCalculation("DBSCAN", "Default", "M2")

