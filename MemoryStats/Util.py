import pandas as pd
from scipy.stats import gmean
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def MemoryConsumptionCalculation(algo, mode, system):

    memory = pd.read_csv("Memory_" + algo + "_" + mode + "_" + system + ".csv")
    time = pd.read_csv("Time_" + algo + "_" + mode + "_" + system + ".csv") 
    
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
    
    time.to_csv("Time_Memory_" + algo + "_" + mode + "_" + system + ".csv")
    
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
    plt.legend(["Default", "Subsampling"])
    plt.xlabel("Points (Rows)")

    if tm == "Memory_Virtual_Max":
        plt.ylabel("Memory (in MB)")
        plt.title(algo + " Memory Usage in " + system)
    else:
        plt.ylabel("Time (in Seconds)")
        plt.title(algo + " Execution Time in " + system)
    
    plt.savefig('Figures/'+tm+'_' + algo + '_' + system +'.pdf', bbox_inches='tight')
    plt.show()


def drawBoxPlot(algo):
    systems = ["M2", "Louise", "Jimmy"]
    modes = ["Default", "SS"]
    
    df_merged = pd.DataFrame()

    
    for s in systems:
        for m in modes:            
            df = pd.read_csv("Time_Memory_" + algo + "_" + m + "_" + s + ".csv")
            df["System"] = s
            df["Mode"] = m
            df_merged = pd.concat([df_merged, df], axis=0)
    df_merged = df_merged.reset_index(drop=True)
    
    sns.boxplot(x = df_merged['System'],
            y = df_merged['TotalTime'],
            hue = df_merged['Mode'])
            # showfliers=False)
    plt.yscale('log')
    
    plt.savefig('Figures/Time_' + algo +'.pdf', bbox_inches='tight')
    
    plt.figure()
    sns.boxplot(x = df_merged['System'],
            y = df_merged['Memory_Virtual_Max'],
            hue = df_merged['Mode'])
            # showfliers=False)
    plt.yscale('log')
    
    plt.savefig('Figures/Memory_' + algo +'.pdf', bbox_inches='tight')

def memoryUsageGraph(algo, mode, system):
    memory = pd.read_csv("Memory_" + algo + "_" + mode + "_" + system + ".csv")
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
        
algo = "AP"
system = "Jimmy"

# mode = "SS"
# MemoryConsumptionCalculation(algo, mode, system)
# mode = "Default"
# MemoryConsumptionCalculation(algo, mode, system)


# drawGraph(algo, system)

memoryUsageGraph(algo, "Default", system)

# drawBoxPlot("DBSCAN")    
    
    
    
    

