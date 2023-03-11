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
from sklearn.covariance import EllipticEnvelope
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
# from memory_profiler import profile

import warnings 
warnings.filterwarnings("ignore")


datasetFolderDir = '../Dataset/Small/'

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
        if df.shape[0] > 100000:
            return True
        
        df = shuffle(df)
        if "target" in df.columns:
            self.y=df["target"].to_numpy()
            self.X=df.drop("target", axis=1)
        elif "outlier" in df.columns:
            self.y=df["outlier"].to_numpy()
            self.X=df.drop("outlier", axis=1)
        else:
            print("Ground Truth not found")
        
        return False
    
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
            t0 = time.time()
            if self.algoName == "OCSVM":
                c = OneClassSVM().fit(self.X)
            elif self.algoName == "LOF":
                c = LocalOutlierFactor(novelty=1).fit(self.X)
            elif self.algoName == "EE":
                c = EllipticEnvelope().fit(self.X)
                
            l = c.predict(self.X)
            t1 = time.time()
            l = [0 if x == 1 else 1 for x in l]
            f1 = (metrics.f1_score(self.y, l))
            
            print("Default--")
            print("F1: ", f1, " and Time: ", t1-t0)
            
        if mode == "optimized":
            if self.bestParams == []:
                print("Calculate the paramters first.")
                return
            t0 = time.time()
            if self.algoName == "OCSVM":
                c = OneClassSVM(kernel=self.bestParams[0], degree=self.bestParams[1], gamma=self.bestParams[2], coef0=self.bestParams[3], tol=self.bestParams[4], nu=self.bestParams[5], 
                                  shrinking=self.bestParams[6], cache_size=self.bestParams[7], max_iter=self.bestParams[8]).fit(self.X)
            elif self.algoName == "LOF":
                c = LocalOutlierFactor(n_neighbors=self.bestParams[0], algorithm=self.bestParams[1], leaf_size=self.bestParams[2], metric=self.bestParams[3], p=self.bestParams[4],
                                       n_jobs=self.bestParams[5], novelty=1).fit(self.X)
            elif self.algoName == "EE":
                c = EllipticEnvelope(assume_centered=self.bestParams[0], support_fraction=self.bestParams[1], contamination=self.bestParams[2]).fit(self.X)
                
            l = c.predict(self.X)
            l = [0 if x == 1 else 1 for x in l]
            f1 = (metrics.f1_score(self.y, l))
            
            t1 = time.time()
            print("Whole dataset with best parameters--")
            print("F1: ", f1, " and Time: ", t1-t0)
        return f1, t1-t0
    
    def determineParam(self, comparison_mode, comparison_mode_algo):
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
                t = threading.Thread(target=self.worker_determineParam, args=(parameters_to_send,self.X_batches[batch_index], self.y_batches[batch_index], batch_index, comparison_mode_algo))
                threads.append(t)
                t.start()
                batch_index += 1
            for t in threads:
                t.join()
            df = pd.read_csv("Output/Rank.csv")
            
            if comparison_mode == "F1T":
                df["W"] = df.Compare/df.Time
            elif comparison_mode == "F1":
                df["W"] = df.Compare
            
            # print(params)
            # print(df)
            h_r = df["W"].idxmax()
            params[1] = params[2][df["Batch"].iloc[h_r]-start_index]
            
        self.bestParams = [p[1] for p in self.parameters]
    
    def worker_determineParam(self, parameter, X, y, batch_index, comparison_mode_algo):        
        t0 = time.time()
        if self.algoName == "OCSVM":
            c = OneClassSVM(kernel=parameter[0], degree=parameter[1], gamma=parameter[2], coef0=parameter[3], tol=parameter[4], nu=parameter[5], 
                          shrinking=parameter[6], cache_size=parameter[7], max_iter=parameter[8]).fit(X)
        elif self.algoName == "LOF":
            c = LocalOutlierFactor(n_neighbors=parameter[0], algorithm=parameter[1], leaf_size=parameter[2], metric=parameter[3], p=parameter[4], 
                                            n_jobs=parameter[5], novelty=1).fit(X)
        elif self.algoName == "EE":
            c = EllipticEnvelope(assume_centered=parameter[0], support_fraction=parameter[1], contamination=parameter[2]).fit(X)
        
        l = c.predict(X)
        
        t1 = time.time()
        cost = t1-t0
    
        l = [0 if x == 1 else 1 for x in l]
        
        f1_comp = self.getF1_Comp(X, l, comparison_mode_algo)

        saveStr = str(batch_index)+","+str(f1_comp)+","+str(cost)+"\n"    
        f = open("Output/Rank.csv", 'a')
        f.write(saveStr)
        f.close()
    
    def getF1_Comp(self, X, l, algo):
        cont = np.mean(l)
        if cont == 0 or cont > 0.5:
            cont = 'auto'
            
        if algo == "LOF":
            lof = LocalOutlierFactor().fit_predict(X)
            lof = [0 if x == 1 else 1 for x in lof]
            f1_lof = metrics.f1_score(l, lof)
            return f1_lof
        elif algo == "IF":
            iforest = IsolationForest().fit(X)
            ifl = iforest.predict(X)    
            ifl = [0 if x == 1 else 1 for x in ifl]
            f1_if = (metrics.f1_score(l, ifl))
            return f1_if
        elif algo == "OCSVM":
            ocsvm = OneClassSVM(nu=cont).fit(X)
            ossvml = ocsvm.predict(X)
            ossvml = [0 if x == 1 else 1 for x in ossvml]
            f1_ocsvm = metrics.f1_score(l, ossvml)
            return f1_ocsvm
        elif algo == "EE":
            ee = EllipticEnvelope().fit(X)
            eel = ee.predict(X)
            eel = [0 if x == 1 else 1 for x in eel]
            f1_ee = metrics.f1_score(l, eel)
            return f1_ee
            
    
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
                c = OneClassSVM(kernel=parameter[0], degree=parameter[1], gamma=parameter[2], coef0=parameter[3], tol=parameter[4], nu=parameter[5], 
                              shrinking=parameter[6], cache_size=parameter[7], max_iter=parameter[8]).fit(X)
            elif self.algoName == "LOF":
                c = LocalOutlierFactor(n_neighbors=parameter[0], algorithm=parameter[1], leaf_size=parameter[2], metric=parameter[3], p=parameter[4], 
                                            n_jobs=parameter[5], novelty=1).fit(X)
            elif self.algoName == "EE":
                c = EllipticEnvelope(assume_centered=parameter[0], support_fraction=parameter[1], contamination=parameter[2]).fit(X)
            l = c.predict(X)
            l = [0 if x == 1 else 1 for x in l]

            with open("Output/Temp/"+str(batch_index)+".csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(y, l))

        if mode == "B":
            ll = []
            for c in self.models:
                ll.append(c.predict(X))

            if self.algoName == "OCSVM":
                c = OneClassSVM(kernel=parameter[0], degree=parameter[1], gamma=parameter[2], coef0=parameter[3], tol=parameter[4], nu=parameter[5], 
                              shrinking=parameter[6], cache_size=parameter[7], max_iter=parameter[8]).fit(X)
            elif self.algoName == "LOF":
                c = LocalOutlierFactor(n_neighbors=parameter[0], algorithm=parameter[1], leaf_size=parameter[2], metric=parameter[3], p=parameter[4], 
                                            n_jobs=parameter[5], novelty=1).fit(X)
            elif self.algoName == "EE":
                c = EllipticEnvelope(assume_centered=parameter[0], support_fraction=parameter[1], contamination=parameter[2]).fit(X)
            
            l = c.predict(X)
            
            l = [x*5 for x in l]
            
            ll.append(l)
            
            self.models.append(c)
            
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
        # print("Accelerated F1: ",f1)
        return f1
    
    def run(self, mode):
        comparison_modes = ["F1", "F1T"]
        comparison_mode_algos = ["LOF", "OCSVM", "IF", "EE"]
        # str_cmodes = "Filename"
        str_values = self.fileName
        for cma in comparison_mode_algos:
            for cm in comparison_modes:
                if cma == self.algoName:
                    str_values=str_values+",0,0"
                    continue
                print(cm, cma)
                t0 = time.time()
                self.subSample(100)
                self.determineParam(cm, cma)
                self.rerun(mode)
                t1 = time.time()
                f1_ss = self.AUL_F1()
                time_ss = t1-t0 
                # str_cmodes=str_cmodes+",F1_"+cm+cma+",Time_"+cm+cma
                
                str_values=str_values+","+str(f1_ss)+","+str(time_ss)
        
        f=open("Stats/"+self.algoName+"_SubsampleAlgoComp.csv", "w")
        f.write(str_values+'\n')
        f.close()
        print(str_values)
        return f1_ss, time_ss
        
    
    
def algo_parameters(algo):
    parameters = []
    if algo == "OCSVM":
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
    
    if algo =="LOF":
        n_neighbors=[2,5,10,20,50,100]
        algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size=[5,10,20,30,50,75,100] 
        # metric=["minkowski", "cityblock", "cosine", "euclidean", "nan_euclidean"]
        metric=["minkowski", "cityblock", "euclidean"]
        p=[3,4]                           
        # contamination=['auto', 0.05, 0.1, 0.2] 
        n_jobs=[None, -1]
        
        parameters.append(["n_neighbors", 20, n_neighbors])
        parameters.append(["algorithm", 'auto', algorithm])        
        parameters.append(["leaf_size", 30, leaf_size])
        parameters.append(["metric", 'minkowski', metric])   
        parameters.append(["p", 2, p])
        # parameters.append(["contamination", "auto", contamination])
        parameters.append(["n_jobs", None, n_jobs])   
    
    elif algo == "EE":
        assume_centered = [True, False]
        support_fraction = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        contamination = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        parameters.append(["assume_centered", False, assume_centered])
        parameters.append(["support_fraction", None, support_fraction])
        parameters.append(["contamination", 0.1, contamination])
    
    return parameters
            
if __name__ == '__main__':
    algorithm = "OCSVM"
    # algorithm = "LOF"
    
    folderpath = datasetFolderDir
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    # if os.path.exists("Stats/"+algorithm+".csv"):
    #     done_files = pd.read_csv("Stats/"+algorithm+".csv")
    #     done_files = done_files["Filename"].to_numpy()
    #     # print(done_files)
    #     master_files = [x for x in master_files if x not in done_files]
    
    master_files.sort()
    # print(master_files)
    
    
    if os.path.exists("Stats/"+algorithm+".csv") == 0:
        f=open("Stats/"+algorithm+".csv", "w")
        f.write('Filename,F1_WD,Time_WD,F1_SS,Time_SS,F1_WO,Time_WO\n')
        f.close()
    
    if os.path.exists("Stats/"+algorithm+"_SubsampleAlgoComp.csv") == 0:
        f=open("Stats/"+algorithm+"_SubsampleAlgoComp.csv", "w")
        f.write('Filename,F1_F1LOF,Time_F1LOF,F1_F1TLOF,Time_F1TLOF,F1_F1OCSVM,Time_F1OCSVM,F1_F1TOCSVM,Time_F1TOCSVM,F1_F1IF,Time_F1IF,F1_F1TIF,Time_F1TIF,F1_F1EE,Time_F1EE,F1_F1TEE,Time_F1TEE\n')
        f.close()
    
    for file in master_files:
        print(file)
        try:
            parameters = algo_parameters(algorithm)
            algoRun = AUL(parameters, file, algorithm)
            # algoRun.readData_arff()
            tooLarge = algoRun.readData()
            if tooLarge:
                continue
            # f1_wd, time_wd = algoRun.runWithoutSubsampling("default")
            f1_ss, time_ss = algoRun.run("B")
            print("Best Parameters: ", algoRun.bestParams)
            # f1_wo, time_wo = algoRun.runWithoutSubsampling("optimized")
            algoRun.destroy()
            
            # # WRITE TO FILE
            # f=open("Stats/"+algorithm+".csv", "a")
            # f.write(file+','+str(f1_wd)+','+str(time_wd)+','+str(f1_ss)+','+str(time_ss)+','+str(f1_wo)+','+str(time_wo) +'\n')
            # f.close()
                
        except:
            print("Fail")
    
        