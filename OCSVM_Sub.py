import sys
import os
import shutil
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
from scipy.io import arff
import threading
from memory_profiler import profile



import warnings 
warnings.filterwarnings("ignore")


datasetFolderDir = '../Dataset/Small/'

# fname = 'coil2000'

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
        
        if os.path.isdir("Output/") == 0:    
            os.mkdir("Output")
            os.mkdir("Output/Temp")
            
    def destroy(self):
        if os.path.isdir("Output"):
            shutil.rmtree("Output")
        
        
    def readData(self):
        df = pd.read_csv(datasetFolderDir+self.fileName+".csv")
    
        df = shuffle(df)
        if "target" in df.columns:
            self.y=df["target"].to_numpy()
            self.X=df.drop("target", axis=1)
        elif "outlier" in df.columns:
            self.y=df["outlier"].to_numpy()
            self.X=df.drop("outlier", axis=1)
        else:
            print("Ground Truth not found")
            
    def readData_arff(self):
        data = arff.loadarff(datasetFolderDir+self.fileName+".arff")
        df = pd.DataFrame(data[0])
        df["outlier"] = df["outlier"].str.decode("utf-8")
        df["outlier"] = pd.Series(np.where(df.outlier.values == "yes", 1, 0),df.index)
        self.y=df["outlier"].to_numpy()
        self.X=df.drop("outlier", axis=1)
        
    def subSample(self, batch_count):
        batch_size = int(len(self.X)/batch_count)
        self.X_batches = [self.X[i:i+batch_size] for i in range(0, len(self.X), batch_size)]
        self.y_batches = [self.y[i:i+batch_size] for i in range(0, len(self.y), batch_size)]
        
    def runWithoutSubsampling(self, mode):
        if mode == "default":
            # self.readData()
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
        return f1, t1-t0
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
                t = threading.Thread(target=self.worker_determineParam, args=(parameters_to_send,self.X_batches[batch_index], self.y_batches[batch_index], batch_index))
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
    
    def worker_determineParam(self, parameter, X, y, batch_index):        
        t0 = time.time()
        if self.algoName == "OCSVM":
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
            
                
    def rerun(self, mode):
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
                t = threading.Thread(target=self.worker_rerun, args=(self.bestParams,self.X_batches[batch_index], self.y_batches[batch_index], batch_index, mode))
                threads.append(t)
                t.start()
                batch_index += 1
            for t in threads:
                t.join()
    
    def worker_rerun(self, parameter, X, y, batch_index, mode):
        if mode == "A":
            if self.algoName == "OCSVM":
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

            if self.algoName == "OCSVM":
                clustering = OneClassSVM(kernel=parameter[0], degree=parameter[1], gamma=parameter[2], coef0=parameter[3], tol=parameter[4], nu=parameter[5], 
                              shrinking=parameter[6], cache_size=parameter[7], max_iter=parameter[8]).fit(X)

            l = clustering.predict(X)

            l = [x*5 for x in l]
            
            ll.append(l)
            
            self.models.append(clustering)
            
            ll = np.array(ll)
            ll = ll.mean(axis=0)
            
            ll = [0 if x > 0 else 1 for x in ll]
            
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
        f1 = metrics.f1_score(yy, ll)
        print("Accelerated F1: ",f1)
        return f1
    
    def run(self, mode):
        
        
        t0 = time.time()
        self.subSample(100)
        self.determineParam()
        self.rerun(mode)
        t1 = time.time()
        f1_ss = self.AUL_F1()
        time_ss = t1-t0
        print("Accelerated Time: ", time_ss)
        return f1_ss, time_ss
        
    
    
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
    
    folderpath = datasetFolderDir
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    if os.path.exists("Stats/"+algorithm+".csv"):
        done_files = pd.read_csv("Stats/"+algorithm+".csv")
        done_files = done_files["Filename"].to_numpy()
        # print(done_files)
    master_files = [x for x in master_files if x not in done_files]
    
    master_files.sort()
    print(master_files)
    
    
    if os.path.exists("Stats/"+algorithm+".csv") == 0: 
        f=open("Stats/"+algorithm+".csv", "w")
        f.write('Filename,F1_WD,Time_WD,F1_SS,Time_SS,F1_WO,Time_WO\n')
        f.close()
    
    for file in master_files:
        print(file)
        try:
            parameters = algo_parameters(algorithm)
            algoRun = AUL(parameters, file, algorithm)
            # algoRun.readData_arff()
            algoRun.readData()
            f1_wd, time_wd = algoRun.runWithoutSubsampling("default")
            f1_ss, time_ss = algoRun.run("B")
            f1_wo, time_wo = algoRun.runWithoutSubsampling("optimized")
            algoRun.destroy()
            
            #WRITE TO FILE
            f=open("Stats/"+algorithm+".csv", "a")
            f.write(file+','+str(f1_wd)+','+str(time_wd)+','+str(f1_ss)+','+str(time_ss)+','+str(f1_wo)+','+str(time_wo) +'\n')
            f.close()
        
        except:
            print("Fail")
        
    
        
        