import sys
import os
import glob
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
from memory_profiler import profile

datasetFolderDir = 'Dataset/'


fname = 'coil2000'

# @profile

class AUL:
    def __init__(self, parameters, fileName, algoName):
        self.parameters = parameters
        self.fileName = fileName
        self.algoName = algoName
        self.X = []
        self.y = []
        self.X_batches = []
        self.y_batches = []
        self.bestParams = []
        self.models = []
        
    def readData(self):    
        df = pd.read_csv(datasetFolderDir+self.fileName+".csv")
    
        df = shuffle(df)
        
        self.y=df["target"].to_numpy()
        self.X=df.drop("target", axis=1)
        
    def subSample(self, batch_count):
        batch_size = int(len(self.X)/batch_count)
        self.X_batches = [self.X[i:i+batch_size] for i in range(0, len(self.X), batch_size)]
        self.y_batches = [self.y[i:i+batch_size] for i in range(0, len(self.y), batch_size)]
        
    def runWithoutSubsampling(self, mode):
        if mode == "default":
            self.readData()
            t0 = time.time()
            c = OneClassSVM().fit(self.X)
            l = c.predict(self.X)
            l = [0 if x == 1 else 1 for x in l]
            
            f1 = (metrics.f1_score(self.y, l))
            
            t1 = time.time()
            print("Default--")
            print("F1: ", f1, " and Time: ", t1-t0)
        
            
        if mode == "optimized":
            if self.bestParams == []:
                print("Calculate the paramters first.")
                return
            t0 = time.time()
            c = OneClassSVM(kernel=self.bestParams[0], degree=self.bestParams[1], gamma=self.bestParams[2], coef0=self.bestParams[3], tol=self.bestParams[4], nu=self.bestParams[5], 
                                  shrinking=self.bestParams[6], cache_size=self.bestParams[7], max_iter=self.bestParams[8]).fit(self.X)
            l = c.predict(self.X)
            l = [0 if x == 1 else 1 for x in l]
            
            f1 = (metrics.f1_score(self.y, l))
            
            t1 = time.time()
            print("Whole dataset with best parameters--")
            print("F1: ", f1, " and Time: ", t1-t0)
    def determineParam(self):
        batch_index = 0
        for params in self.parameters:
            threads = []
            f = open("Output/Rank.csv", 'w')
            f.write("Batch,Compare,Time\n")
            f.close()
            start_index = batch_index
            for p_v in params[2]:
                params[1] = p_v
                parameters_to_send = [p[1] for p in self.parameters]
                t = threading.Thread(target=self.worker, args=(parameters_to_send,self.X_batches[batch_index], self.y_batches[batch_index], batch_index, "D"))
                threads.append(t)
                t.start()
                batch_index += 1
            for t in threads:
                t.join()
            
            df = pd.read_csv("Output/Rank.csv")
    
            df["W"] = df.Compare/df.Time
            
            h_r = df["W"].idxmax()
            params[1] = params[2][df["Batch"].iloc[h_r]-start_index]
            
        self.bestParams = [p[1] for p in self.parameters]
    
    def rerun(self):
        if self.bestParams == []:
            print("Determine best parameters before this step.")
            return
        batch_count = 50
        self.subSample(batch_count)
        threads = []
        batch_index = 0
        
        for _ in range(5):
            for _ in range(10):
                if batch_index >= batch_count-1:
                    break
                t = threading.Thread(target=self.worker, args=(self.bestParams,self.X_batches[batch_index], self.y_batches[batch_index], batch_index, "B"))
                threads.append(t)
                t.start()
                batch_index += 1
            for t in threads:
                t.join()
    
    def worker(self, parameter, X, y, batch_index, mode):
        if self.algoName == "OCSVM":
            if mode == "D":
                t0 = time.time()
                clustering = OneClassSVM(kernel=parameter[0], degree=parameter[1], gamma=parameter[2], coef0=parameter[3], tol=parameter[4], nu=parameter[5], 
                                  shrinking=parameter[6], cache_size=parameter[7], max_iter=parameter[8]).fit(X)
                t1 = time.time()
                cost = t1-t0
            
                l = clustering.predict(X)
                l = [0 if x == 1 else 1 for x in l]
                lof = LocalOutlierFactor(n_neighbors=2).fit_predict(X)
                lof = [0 if x == 1 else 1 for x in lof]
                
                # iforest = IsolationForest().fit(X)
                # ifl = iforest.predict(X)    
                # ifl = [0 if x == 1 else 1 for x in ifl]
                
                # f1 = (metrics.f1_score(y, l))
                f1_lof = metrics.f1_score(y, lof)
                # f1_if = (metrics.f1_score(y, ifl))
                
                saveStr = str(batch_index)+","+str(f1_lof)+","+str(cost)+"\n"    
                f = open("Output/Rank.csv", 'a')
                f.write(saveStr)
                f.close()
        
            if mode == "C":
                clustering = OneClassSVM(kernel=parameter[0], degree=parameter[1], gamma=parameter[2], coef0=parameter[3], tol=parameter[4], nu=parameter[5], 
                                  shrinking=parameter[6], cache_size=parameter[7], max_iter=parameter[8]).fit(X)
                l = clustering.predict(X)
                l = [0 if x == 1 else 1 for x in l]

                with open("Output/Temp/"+str(batch_index)+".csv", 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(y, l))

            if mode == "B":
                ll = []
                for c in self.models:
                    ll.append(c.predict(X))
                    
                clustering = OneClassSVM(kernel=parameter[0], degree=parameter[1], gamma=parameter[2], coef0=parameter[3], tol=parameter[4], nu=parameter[5], 
                                  shrinking=parameter[6], cache_size=parameter[7], max_iter=parameter[8]).fit(X)
                l = clustering.predict(X)
                # l = [0 if x == 1 else 1 for x in l]

                l = [x*5 for x in l]
                
                ll.append(l)
                
                self.models.append(clustering)
                
                ll = np.array(ll)
                ll = ll.mean(axis=0)
                
                ll = [0 if x > 0 else 1 for x in ll]
                
                # print("Models: ", len(self.models), metrics.f1_score(lll, l))
                
                with open("Output/Temp/"+str(batch_index)+".csv", 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(y, ll))
                    
    
    def AUL_F1(self):
        df_csv_append = pd.DataFrame()
        csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
        for file in csv_files:
            df = pd.read_csv(file, header=None)
            df_csv_append = pd.concat([df_csv_append, df])
            # df_csv_append = df_csv_append.append(df, ignore_index=True)
    
        yy = df_csv_append[0].tolist()
        ll = df_csv_append[1].tolist()
        print("Accelerated F1: ",metrics.f1_score(yy, ll))
    
    def run(self):
        self.readData()
        
        t0 = time.time()
        self.subSample(100)
        self.determineParam()
        self.rerun()
        t1 = time.time()
        self.AUL_F1()
        print("Accelerated Time: ", t1-t0)
        
        
    
    
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

    algoRun = AUL(parameters, fname, algorithm)
    
    algoRun.runWithoutSubsampling("default")
    
    algoRun.run()
    
    algoRun.runWithoutSubsampling("optimized")
    
        
        