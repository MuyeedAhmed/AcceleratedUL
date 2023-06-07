import pandas as pd
from scipy.stats import gmean
import numpy as np
import sys
import matplotlib.pyplot as plt

def MemoryConsumptionCalculation(algo, mode, system):

    memory = pd.read_csv("Memory_" + algo + "_" + mode + "_" + system + ".csv")
    time = pd.read_csv("Time_" + algo + "_" + mode + "_" + system + ".csv") 
    
    time["TotalTime"] = time["EndTime"] - time["StartTime"]
    
    time["Memory_Median"] = None
    time["Memory_Physical_Median"] = None
    time["Memory_Virtual_Median"] = None
    
    time["Memory_Max"] = None
    time["Memory_Physical_Max"] = None
    time["Memory_Virtual_Max"] = None
    
    
    for index, row in time.iterrows():
        # row["Memory_Median"] = index
        t = memory[(memory["Time"] > row["StartTime"]) & (memory["Time"] < row["EndTime"])]
        if t.empty:
            continue
        memory_physical = t["Memory_Physical"].to_numpy()
        mp_median = np.median(memory_physical)
        mp_max = np.max(memory_physical)
        
        memory_virtual = t["Memory_Virtual"].to_numpy()
        mv_median = np.median(memory_virtual)
        mv_max = np.max(memory_virtual)
        
        time.loc[index, "Memory_Median"] = mp_median + mv_median
        time.loc[index, "Memory_Physical_Median"] = mp_median
        time.loc[index, "Memory_Virtual_Median"] = mv_median
        
        time.loc[index, "Memory_Max"] = mp_max + mv_max
        time.loc[index, "Memory_Physical_Max"] = mp_max
        time.loc[index, "Memory_Virtual_Max"] = int(mv_max)
    
    time.to_csv("Time_Memory_" + algo + "_" + mode + "_" + system + ".csv")
    
    # table = time.pivot(index='Row', columns='Columm', values='Memory_Max')
    # table.to_csv("Max_Memory_Usage_" + algo + "_" + mode + "_" + system + ".csv")

def drawGraph(algo, system):
    default = pd.read_csv("Time_Memory_" + algo + "_Default_" + system + ".csv")
    ss = pd.read_csv("Time_Memory_" + algo + "_SS_" + system + ".csv")

    draw(default, ss, "Memory_Max")
    draw(default, ss, "TotalTime")
    
def draw(df_d, df_s, tm):    
    x = df_s["Row"]
    y_Default = df_d[tm]
    y_SS = df_s[tm]
    
    plt.figure(0)
    print(len(x))
    plt.plot(x[0:len(y_Default)],y_Default, ".",color="red")
    plt.plot(x[0:len(y_SS)],y_SS, ".",color="blue")
        
    plt.grid(True)
    plt.legend(["Default - 10 Features", "Default - 100 Features", "Subsample - 10 Features", "Subsample - 100 Features"])
    plt.xlabel("Points (Rows)")
    if tm == "Memory_Max":
        plt.ylabel("Memory (in MB)")
        plt.title(algo + " Memory Usage in " + system)
    else:
        plt.ylabel("Time (in Seconds)")
        plt.title(algo + " Execution Time in " + system)
    
    plt.savefig('Figures/'+tm+'_' + algo + '_' + system +'.pdf', bbox_inches='tight')
    plt.show()

    
# algo = sys.argv[1]
# mode = sys.argv[2]
# system = sys.argv[3]

algo = "DBSCAN"
system = "Jimmy"

mode = "SS"
MemoryConsumptionCalculation(algo, mode, system)
mode = "Default"
MemoryConsumptionCalculation(algo, mode, system)


drawGraph(algo, system)

