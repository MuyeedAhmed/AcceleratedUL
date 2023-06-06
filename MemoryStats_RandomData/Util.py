import pandas as pd
from scipy.stats import gmean
import numpy as np
import sys
import matplotlib.pyplot as plt

def MemoryConsumptionCalculation(algo, mode, system):

    memory = pd.read_csv("Memory_" + algo + "_" + mode + "_" + system + ".csv")
    time = pd.read_csv("Time_" + algo + "_" + mode + "_" + system + ".csv") 
    
    time["TotalTime"] = time["EndTime"] - time["StartTime"]
    timeTable = time.pivot(index='Row', columns='Columm', values='TotalTime')
    timeTable.to_csv("TimeTable_" + algo + "_" + mode + "_" + system + ".csv")
    
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
    
    
    table = time.pivot(index='Row', columns='Columm', values='Memory_Max')
    table.to_csv("Max_Memory_Usage_" + algo + "_" + mode + "_" + system + ".csv")
    print(table)

def drawGraph(algo, system):
    memoryTable_D = pd.read_csv("Max_Memory_Usage_" + algo + "_Default_" + system + ".csv")
    memoryTable_S = pd.read_csv("Max_Memory_Usage_" + algo + "_SS_" + system + ".csv")

    timeTable_D = pd.read_csv("TimeTable_" + algo + "_Default_" + system + ".csv") 
    timeTable_S = pd.read_csv("TimeTable_" + algo + "_SS_" + system + ".csv") 
    
    draw(memoryTable_D, memoryTable_S, "Memory")
    draw(timeTable_D, timeTable_S, "Time")
    
def draw(df_d, df_s, tm):    
    x = df_s["Row"]
    y_10_Default = df_d["10"]
    y_100_Default = df_d["100"]
    y_10_SS = df_s["10"]
    y_100_SS = df_s["100"]
    
    plt.figure(0)
    print(len(x))
    plt.plot(x[0:len(y_10_Default)],y_10_Default,color="orange")
    plt.plot(x[0:len(y_100_Default)],y_100_Default,color="red")
    plt.plot(x[0:len(y_10_SS)],y_10_SS,color="blue")
    plt.plot(x[0:len(y_100_SS)],y_100_SS,color="navy")
        
    plt.grid(True)
    plt.legend(["Default - 10 Features", "Default - 100 Features", "Subsample - 10 Features", "Subsample - 100 Features"])
    plt.xlabel("Points (Rows)")
    if tm == "Memory":
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
system = "Thelma"

mode = "SS"
MemoryConsumptionCalculation(algo, mode, system)
mode = "Default"
MemoryConsumptionCalculation(algo, mode, system)


drawGraph(algo, system)

