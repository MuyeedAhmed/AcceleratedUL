import sys
import os
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import bisect 
from random import sample
import time
from sklearn.utils import shuffle
import csv

import threading

datasetFolderDir = 'Dataset/'


fname = 'coil2000'

def ocsvm(filename, parameters, parameter_iteration):
    folderpath = datasetFolderDir
    parameters_this_file = deepcopy(parameters)
    
    X = pd.read_csv(folderpath+filename+".csv")
    
    X = shuffle(X)
    
    gt=X["target"].to_numpy()
    X=X.drop("target", axis=1)
    
    
    
    # '''Default F1 and time'''
    # t0 = time.time()
    # c = OneClassSVM().fit(X)
    # l = c.predict(X)
    # l = [0 if x == 1 else 1 for x in l]
    
    # f1 = (metrics.f1_score(gt, l))

    # t1 = time.time()
    # print("F1: ", f1, " and Time: ", t1-t0)
    # ''''''
    
    batch_size = 100
    X_batches = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]
    gt_batches = [gt[i:i+batch_size] for i in range(0, len(gt), batch_size)]
    
    batch_index = 0
    
    t0 = time.time()
    
    for params in parameters_this_file:
        threads = []
        f = open("Output/Rank.csv", 'w')
        f.write("Batch,F1_LOF,Time\n")
        f.close()
        start_index = batch_index
        for p_v in params[2]:
            params[1] = p_v
            parameters_to_send = [p[1] for p in parameters_this_file]
            t = threading.Thread(target=worker, args=(parameters_to_send,X_batches[batch_index], gt_batches[batch_index], batch_index))
            threads.append(t)
            t.start()
            batch_index += 1
        for t in threads:
            t.join()
        
        df = pd.read_csv("Output/Rank.csv")

        # print(df)
        df["W"] = df.F1_LOF/df.Time
        
        h_r = df["W"].idxmax()
        # print(df)
        # print(df["Batch"].iloc[h_r], start_index)
        params[1] = params[2][df["Batch"].iloc[h_r]-start_index]
        
    parameters_to_send = [p[1] for p in parameters_this_file]
    print(parameters_to_send)
    threads = []
    while True:
        if batch_index >= batch_size-1:
            break
        t = threading.Thread(target=worker, args=(parameters_to_send,X_batches[batch_index], gt_batches[batch_index], batch_index))
        threads.append(t)
        t.start()
        batch_index += 1
    for t in threads:
        t.join()
    t1 = time.time()
    
    print("Time: ", t1-t0)
    
    
    '''
    Merge and Calculate
    '''
    import glob
    df_csv_append = pd.DataFrame()
    csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        df_csv_append = pd.concat([df_csv_append, df])
        # df_csv_append = df_csv_append.append(df, ignore_index=True)

    yy = df_csv_append[0].tolist()
    ll = df_csv_append[1].tolist()
    print(metrics.f1_score(yy, ll))
    
    
        
def worker(parameter, X, y, batch_index):
    
    # print("batch_index", batch_index)
    # print(parameters_this_file)
    t0 = time.time()
    clustering = OneClassSVM(kernel=parameter[0], degree=parameter[1], gamma=parameter[2], coef0=parameter[3], tol=parameter[4], nu=parameter[5], 
                          shrinking=parameter[6], cache_size=parameter[7], max_iter=parameter[8]).fit(X)
    t1 = time.time()
    cost = t1-t0
    
    l = clustering.predict(X)
    l = [0 if x == 1 else 1 for x in l]

    with open("Output/Temp/"+str(batch_index)+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(y, l))
    
    
        
    # lof = LocalOutlierFactor(n_neighbors=2).fit_predict(X)
    # lof = [0 if x == 1 else 1 for x in lof]
    
    # iforest = IsolationForest().fit(X)
    # ifl = iforest.predict(X)    
    # ifl = [0 if x == 1 else 1 for x in ifl]
    
    
    f1 = (metrics.f1_score(y, l))
    # f1_lof = (metrics.f1_score(y, lof))
    # f1_if = (metrics.f1_score(y, ifl))

    saveStr = str(batch_index)+","+str(f1)+","+str(cost)+"\n"    
    f = open("Output/Rank.csv", 'a')
    f.write(saveStr)
    f.close()
    
    
    # s = "Batch " + str(batch_index) + ": " + str(f1) + " || LOF: "+str(f1_lof) + " || IF: "+str(f1_if) + "\n"
    # print(s)
    
   
def runAlgo(filename, X, gt, params, parameter_iteration):
    labels = []
    f1 = []
    ari = []
    global withGT
    for i in range(10):
        clustering = OneClassSVM(kernel=params[0][1], degree=params[1][1], gamma=params[2][1], coef0=params[3][1], tol=params[4][1], nu=params[5][1], 
                              shrinking=params[6][1], cache_size=params[7][1], max_iter=params[8][1]).fit(X)
    
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
        if withGT:
            f1.append(metrics.f1_score(gt, l))
        
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
          ari.append(adjusted_rand_score(labels[i], labels[j]))      
    if withGT:
        return np.mean(f1), np.mean(ari)
    else:
        return -1, np.mean(ari) 

    
def algo_parameters(algo):
    if algo == "OCSVM":
        parameters = []
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        degree = [3, 4, 5, 6] # Kernel poly only
        gamma = ['scale', 'auto'] # Kernel ‘rbf’, ‘poly’ and ‘sigmoid’
        coef0 = [0.0, 0.1, 0.2, 0.3, 0.4] # Kernel ‘poly’ and ‘sigmoid’
        tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        shrinking = [True, False]
        cache_size = [50, 100, 200, 400]
        max_iter = [50, 100, 150, 200, 250, 300, -1]
        
        parameters.append(["kernel", 'rbf', kernel])
        parameters.append(["degree", 3, degree])
        parameters.append(["gamma", 'scale', gamma])
        parameters.append(["coef0", 0.0, coef0])
        parameters.append(["tol", 0.001, tol])
        parameters.append(["nu", 0.5, nu])
        parameters.append(["shrinking", True, shrinking])
        parameters.append(["cache_size", 200, cache_size])
        parameters.append(["max_iter", -1, max_iter])
        return parameters
        
        
if __name__ == '__main__':
    algorithm = "OCSVM"

    parameters = algo_parameters(algorithm)

    ocsvm(fname, parameters, algorithm)
    
        
        