import pandas as pd
from scipy.stats import gmean
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def MemoryConsumptionCalculation(algo, mode, system):

    memory = pd.read_csv(algo + "/Memory_" + algo + "_" + mode + "_" + system + ".csv")
    time = pd.read_csv(algo + "/Time_" + algo + "_" + mode + "_" + system + ".csv") 
    
    time = time[time["Completed"] != -1]
    
    time["TotalTime"] = time["EndTime"] - time["StartTime"]
    time['TotalTime'] = time['TotalTime'].mask(time['TotalTime'] < 0, np.nan)

    # time["Memory_Median"] = None
    time["Memory_Physical_Median"] = None
    time["Memory_Virtual_Median"] = None
    
    # time["Memory_Max"] = None
    time["Memory_Physical_Max"] = None
    time["Memory_Virtual_Max"] = None
    
    
    for index, row in time.iterrows():
        # row["Memory_Median"] = index
        # if row["Completed"] == -1:
        #     continue
        if row["Completed"] == -22:
            # print(row["Filename"])
            t = memory[memory["Filename"] == row["Filename"]]
            # print(t)
        else:
            t = memory[(memory["Time"] > row["StartTime"]) & (memory["Time"] < row["EndTime"])]
        if t.empty:
            print(row["Filename"])
            continue
        memory_physical = t["Memory_Physical"].to_numpy()
        mp_median = np.median(memory_physical)
        mp_max = np.max(memory_physical)
        
        memory_virtual = t["Memory_Virtual"].to_numpy()
        mv_median = np.median(memory_virtual)
        mv_max = np.max(memory_virtual)
        
        # time.loc[index, "Memory_Median"] = mp_median + mv_median
        time.loc[index, "Memory_Physical_Median"] = mp_median
        time.loc[index, "Memory_Virtual_Median"] = mv_median
        
        # time.loc[index, "Memory_Max"] = mp_max + mv_max
        time.loc[index, "Memory_Physical_Max"] = mp_max
        time.loc[index, "Memory_Virtual_Max"] = int(mv_max)
    
    
    time.to_csv(algo + "/Time_Memory_" + algo + "_" + mode + "_" + system + "_All.csv", index = False)

    time.drop_duplicates(subset='Filename', keep='first', inplace=True)
    
    df_SS = pd.read_csv("FileList.csv")
    ss_files = df_SS["Filename"].to_numpy()
    
    filtered_df = time[time['Filename'].isin(ss_files)]
    filtered_df.to_csv(algo + "/Time_Memory_" + algo + "_" + mode + "_" + system + ".csv", index = False)
        
    # table = time.pivot(index='Row', columns='Columm', values='Memory_Max')
    # table.to_csv("Max_Memory_Usage_" + algo + "_" + mode + "_" + system + ".csv")

def drawGraph(algo, system):
    default = pd.read_csv("Time_Memory_" + algo + "_Default_" + system + ".csv")
    ss = pd.read_csv("Time_Memory_" + algo + "_SS_" + system + ".csv")

    draw(default, ss, "Memory_Virtual_Max", algo, system)
    draw(default, ss, "TotalTime", algo, system)
    
def draw(df_d, df_s, tm, algo, system):    
    df_s = df_s[df_s['Filename'].isin(df_d['Filename'])]
    df_d = df_d[df_d['Filename'].isin(df_s['Filename'])]
    
    x_Default = df_d["Row"]
    x_SS = df_s["Row"]
    
    y_Default = df_d[tm]
    y_SS = df_s[tm]
    
    plt.figure(0)
    plt.plot(x_Default,y_Default, ".",color="red")
    plt.plot(x_SS,y_SS, ".",color="blue")
        
    plt.grid(True)
    plt.legend(["Default", "SAC"])
    plt.xlabel("Points (Rows)")

    if tm == "Memory_Virtual_Max":
        plt.ylabel("Memory (in MB)")
        plt.title(algo + " Memory Usage in " + system)
    else:
        plt.ylabel("Time (in Seconds)")
        plt.title(algo + " Execution Time in " + system)
    
    # plt.savefig('Figures/'+tm+'_' + algo + '_' + system +'.pdf', bbox_inches='tight')
    plt.show()



def drawBoxPlot(algo):
    systems = ["M2", "Louise", "Jimmy", "Thelma"]
    modes = ["Default", "SS"]
    
    df_merged = pd.DataFrame()

    
    for s in systems:
        for m in modes:            
            df = pd.read_csv(algo + "/Time_Memory_" + algo + "_" + m + "_" + s + "_All.csv")
            df = df.dropna()

            df["System"] = s
            df["Mode"] = m
            if df_merged.empty == 0:
                
                common_names = set(df_merged['Filename']).intersection(df['Filename'])
                print(len(common_names))
                df = df[df['Filename'].isin(common_names)]
                df_merged = df_merged[df_merged['Filename'].isin(common_names)]
            df_merged = pd.concat([df_merged, df], axis=0)
    df_merged = df_merged.reset_index(drop=True)
    
    df_merged['Mode'] = df_merged['Mode'].replace('SS', 'SAC')
    
    ''' Grouped Results'''    
    grouped = df_merged.groupby(['System', 'Mode'])
    result = grouped['TotalTime'].agg(['min', 'mean', 'max', 'median'])
    print(result)
    
    '''Plot Time '''
    
    f1 = plt.figure(figsize=(8, 6))

    f1 = sns.boxplot(x = df_merged['System'],
            y = df_merged['TotalTime'],
            hue = df_merged['Mode'],
            # palette="Blues",
            flierprops={'marker': 'o','markerfacecolor': 'none', 'markeredgecolor': 'black', 'markersize': 6})
            # showfliers=False)
    plt.yscale('log')
    
    
    
    plt.xlabel("")
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.yticks(fontsize=14)
    f1.set_xticklabels(["Sys1","Sys2", "Sys3", "Sys4"], fontsize=14)
    f1.legend(fontsize=14)
    plt.savefig('Figures/Time_' + algo +'.pdf', bbox_inches='tight')
    
    '''Plot Memory'''
    f2 = plt.figure(figsize=(8, 6))
    f2 = sns.boxplot(x = df_merged['System'],
            y = df_merged['Memory_Virtual_Max'],
            hue = df_merged['Mode'],
            flierprops={'marker': 'o','markerfacecolor': 'none', 'markeredgecolor': 'black', 'markersize': 6})
            # showfliers=False)
    plt.yscale('log')
    
    plt.xlabel("")
    plt.ylabel("Maximum Memory Usage (MB)", fontsize=14)
    plt.yticks(fontsize=14)
    f2.legend(fontsize=14)

    f2.set_xticklabels(["Sys1","Sys2", "Sys3", "Sys4"], fontsize=14)
    
    plt.savefig('Figures/Memory_' + algo +'.pdf', bbox_inches='tight')
    

def memoryUsageGraph(algo, mode, system):
    memory = pd.read_csv(algo +"/Memory_" + algo + "_" + mode + "_" + system + ".csv")
    Filenames = memory["Filename"].to_numpy()
    files = set(Filenames)
    # np.set_printoptions(threshold=1000000)
    
    plt.figure(0)
    for file in files:
        df = memory[memory["Filename"] == file]
        print(df.shape[0], end=' ')
        
        if df.shape[0] < 50:
            continue
        print(df["Filename"])
        print(df["Time"].to_numpy())
        df["Time"] = df["Time"] - df["Time"].min()
        df = df.sort_values(by='Time')
        
        t = df["Time"].to_numpy()
        m = df["Memory_Virtual"].to_numpy()
        
        rolling_average = df['Memory_Virtual'].rolling(window=10, min_periods=1).mean()

        # print(t)
        # break
        
        plt.plot(t, rolling_average)
        plt.axvline(x = 7200, color = 'r', linestyle = '-')
        plt.axvline(x = 1800, color = 'b', linestyle = '-')
        plt.xlim([0, 30000])

def RunStatusDefault(algo, mode, system):
    time = pd.read_csv(algo+"/Time_" + algo + "_" + mode + "_" + system + ".csv") 
    
    count = time['Completed'].value_counts()
    print(count)
    
    time = time[time["Completed"] != -1]
    
    
    
    df_SS = pd.read_csv("FileList.csv")
    ss_files = df_SS["Filename"].to_numpy()
    
    filtered_time = time[time['Filename'].isin(ss_files)]
    
    count = filtered_time['Completed'].value_counts()
    print(count)
    
    time_out = 0
    mem_out = 0
    done = 0
    for sf in ss_files:
        df = time[time["Filename"] == sf]
        if df.shape[0] == 0:
            continue
        if (df["Completed"] == 1).any():
            done+=1
        elif (df["Completed"] == -23).any():
            time_out+=1
        else:
            mem_out+=1
    print(done, mem_out, time_out)
# algo = "DBSCAN"
# system = "Thelma"

# mode = "SS"
# MemoryConsumptionCalculation(algo, mode, system)
# mode = "Default"
# MemoryConsumptionCalculation(algo, mode, system)

# MemoryConsumptionCalculation("DBSCAN", "SS", "M2")
# MemoryConsumptionCalculation("DBSCAN", "SS", "Jimmy")
# MemoryConsumptionCalculation("DBSCAN", "SS", "Thelma")
# MemoryConsumptionCalculation("DBSCAN", "SS", "Louise")

# MemoryConsumptionCalculation("DBSCAN", "Default", "M2")
# MemoryConsumptionCalculation("DBSCAN", "Default", "Jimmy")
# MemoryConsumptionCalculation("DBSCAN", "Default", "Thelma")
# MemoryConsumptionCalculation("DBSCAN", "Default", "Louise")


# drawGraph(algo, system)

# memoryUsageGraph(algo, "Default", system)

drawBoxPlot("DBSCAN")    
    
# RunStatusDefault("DBSCAN", "Default", "Thelma")
    


