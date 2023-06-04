import pandas as pd
from scipy.stats import gmean
import numpy as np
import sys


def MemoryConsumptionCalculation(algo, mode, system):

    memory = pd.read_csv("Memory_" + algo + "_" + mode + "_" + system + ".csv")
    time = pd.read_csv("Time_" + algo + "_" + mode + "_" + system + ".csv") 
    
    
    
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

algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

# algo = "DBSCAN"
# mode = "SS"
# system = "Louise"


MemoryConsumptionCalculation(algo, mode, system)


