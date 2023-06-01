import pandas as pd
from scipy.stats import gmean
import numpy as np


time = pd.read_csv("Time.csv")
run = pd.read_csv("Run.csv") 

run["Memory_Median"] = None
run["Memory_Physical_Median"] = None
run["Memory_Virtual_Median"] = None

run["Memory_Max"] = None
run["Memory_Physical_Max"] = None
run["Memory_Virtual_Max"] = None


for index, row in run.iterrows():
    # row["Memory_Median"] = index
    t = time[(time["Time"] > row["StartTime"]) & (time["Time"] < row["EndTime"])]
    if t.empty:
        continue
    memory_physical = t["Memory_Physical"].to_numpy()
    mp_median = np.median(memory_physical)
    mp_max = np.max(memory_physical)
    
    memory_virtual = t["Memory_Virtual"].to_numpy()
    mv_median = np.median(memory_virtual)
    mv_max = np.max(memory_virtual)
    
    run.loc[index, "Memory_Median"] = mp_median + mv_median
    run.loc[index, "Memory_Physical_Median"] = mp_median
    run.loc[index, "Memory_Virtual_Median"] = mv_median
    
    run.loc[index, "Memory_Max"] = mp_max + mv_max
    run.loc[index, "Memory_Physical_Max"] = mp_max
    run.loc[index, "Memory_Virtual_Max"] = int(mv_max)



print(run)
print(run.iloc[0])

table = run.pivot(index='Row', columns='Columm', values='Memory_Max')

table.to_csv("Memory_Max.csv")
print(table)
