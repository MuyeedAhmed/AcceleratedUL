import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn import metrics

import time

# from memory_profiler import profile
from timeout import timeout

@timeout(500)
def hue():
    
    datasetFolderDir = '../Dataset/Data_Small/'
    datasetFolderDir = "Temp/"
    
    fileName = "analcatdata_challenger"
    df = pd.read_csv(datasetFolderDir+fileName+".csv")
    
    if "target" in df.columns:
        y=df["target"].to_numpy()
        X=df.drop("target", axis=1)
    elif "outlier" in df.columns:
        y=df["outlier"].to_numpy()
        X=df.drop("outlier", axis=1)
    else:
        print("Ground Truth not found")
    # print(X)
    
    
    t0 = time.time()
    # print(t0)
    c = OneClassSVM(nu=0.1, verbose=True).fit(X)
    l = c.predict(X)
    l = [0 if x == 1 else 1 for x in l]
    
    f1 = (metrics.f1_score(y, l))
    
    t1 = time.time()
    print("Default--")
    print("F1: ", f1, " and Time: ", t1-t0)
    
    
        
    
        
hue()