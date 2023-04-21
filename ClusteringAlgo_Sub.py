import os
import shutil
import glob
import pandas as pd
import numpy as np
# from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
import time
from sklearn.utils import shuffle
import threading
# from memory_profiler import profile
import warnings 
warnings.filterwarnings("ignore")
import itertools
from sklearn.metrics import f1_score
import multiprocessing


# from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM
# from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# datasetFolderDir = '/jimmy/hdd/ma234/Dataset/'
# datasetFolderDir = 'Temp/'
datasetFolderDir = '/Users/muyeedahmed/Desktop/Research/Dataset/'

class AUL_Clustering:
    def __init__(self, parameters, fileName, algoName, n_cluster=2):
        self.parameters = parameters
        self.fileName = fileName
        self.algoName = algoName
        self.X = []
        self.y = []
        self.X_batches = []
        self.y_batches = []
        self.bestParams = []
        self.models = []
        
        self.n_cluster = n_cluster
        
        self.determine_param_mode = "ARI"
        self.determine_param_clustering_algo = "KM"
        self.rerun_mode = "A"
        self.AD_algo_merge = "LOF"
        
        
        if os.path.isdir("Output/") == 0:    
            os.mkdir("Output")
            os.mkdir("Output/Temp")
            
    def destroy(self):
        if os.path.isdir("Output"):
            shutil.rmtree("Output")
        
        
    def readData(self):
        try:
            df = pd.read_csv(datasetFolderDir+self.fileName+".csv")
        except:
            print("File Doesn't Exist!")
            return True, 0, 0
        # if df.shape[0] < 10000: #Skip if dataset contains less than 10,000 rows
        #     return True, 0, 0
        if df.shape[0] > 10000: #Skip if dataset contains less than 10,000 rows
            return True, 0, 0
        
        
        df = shuffle(df)
        if "target" in df.columns:
            self.y=df["target"].to_numpy()
            self.X=df.drop("target", axis=1)
        elif "class" in df.columns:
            self.y=df["class"].to_numpy()
            self.X=df.drop("class", axis=1)
        else:
            print("Ground Truth not found. Please set the ground truth as \"class\" or \"target\" ")
            
        # print(self.fileName, df.shape, (os.path.getsize(datasetFolderDir+self.fileName+".csv") >> 20))
        
        return False, df.shape, os.path.getsize(datasetFolderDir+self.fileName+".csv")
    
    def subSample(self, batch_count):
        batch_size = int(len(self.X)/batch_count)
        # print(batch_size)
        self.X_batches = [self.X[i:i+batch_size] for i in range(0, len(self.X), batch_size)]
        self.y_batches = [self.y[i:i+batch_size] for i in range(0, len(self.y), batch_size)]
    
    def runWithoutSubsampling(self, mode):
        df = pd.read_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+".csv")
        
        self.y=df["y"].to_numpy()
        self.X=df.drop("y", axis=1)
        
        t0 = time.time()
        p = multiprocessing.Process(target=self.runWithoutSubsampling_w, args=(mode,))
        p.start()
        p.join(timeout=7200)
        t1 = time.time()
        
        if p.is_alive():
            p.terminate()
            print("Terminated")
            return 0, 0, 7200
        else:
            if os.path.exists("ClusteringOutput/"+self.fileName+"_"+self.algoName+"_WDL.csv") == 0:
                print("Error in completion")
                return -1,-1,-1
            df = pd.read_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+"_WDL.csv")
            ari = adjusted_rand_score(df["Default_labels"], df["l"])
            ari_wd = adjusted_rand_score(df["y"], df["Default_labels"])
            return ari, ari_wd, t1-t0
    
    def runWithoutSubsampling_w(self, mode):
        l_ss = self.X["l"].to_numpy()
        self.X = self.X.drop("l", axis=1)
        if mode == "default":
            # t0 = time.time()
            
            if self.algoName == "AP":
                c = AffinityPropagation().fit(self.X)
            elif self.algoName == "SC":
                c = SpectralClustering(n_clusters=self.n_cluster).fit(self.X)
            
                
            l = c.labels_
            # t1 = time.time()
            # ari = adjusted_rand_score(self.y, l)
            # print("Default--")
            # print("ARI: ", ari, " and Time: ", t1-t0)
            self.X["y"] = self.y
            self.X["l"] = l_ss
            self.X["Default_labels"] = l
            self.X.to_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+"_WDL.csv")
            
        if mode == "optimized":
            if self.bestParams == []:
                print("Calculate the paramters first.")
                return
            # t0 = time.time()
            
            if self.algoName == "AP":
                c = AffinityPropagation(damping=self.bestParams[0], max_iter=self.bestParams[1], convergence_iter=self.bestParams[2]).fit(self.X)
            elif self.algoName == "SC":
                c = SpectralClustering(n_clusters=self.n_cluster, eigen_solver=self.bestParams[0], n_components=self.bestParams[1], 
                                       n_init=self.bestParams[2], gamma=self.bestParams[3], affinity=self.bestParams[4], 
                                       n_neighbors=self.bestParams[5], assign_labels=self.bestParams[6], 
                                       degree=self.bestParams[7], n_jobs=self.bestParams[8]).fit(self.X)
            
            l = c.labels_
            # t1 = time.time()
            # ari = adjusted_rand_score(self.y, l)
            # print("Whole dataset with best parameters--")
            # print("ARI: ", ari, " and Time: ", t1-t0)
        
        
        # return ari, t1-t0
    
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
            
            if self.determine_param_mode == "ARI_T":
                df["W"] = df.Compare/df.Time
            elif self.determine_param_mode == "ARI":
                df["W"] = df.Compare
            
            h_r = df["W"].idxmax()
            params[1] = params[2][df["Batch"].iloc[h_r]-start_index]
            
        self.bestParams = [p[1] for p in self.parameters]
    
    def worker_determineParam(self, parameter, X, y, batch_index):        
        t0 = time.time()
        if self.algoName == "AP":
            c = AffinityPropagation(damping=parameter[0], max_iter=parameter[1], convergence_iter=parameter[2]).fit(X)
        elif self.algoName == "SC":
            c = SpectralClustering(n_clusters=self.n_cluster, eigen_solver=parameter[0], n_components=parameter[1], 
                                   n_init=parameter[2], gamma=parameter[3], affinity=parameter[4], 
                                   n_neighbors=parameter[5], assign_labels=parameter[6], 
                                   degree=parameter[7], n_jobs=parameter[8]).fit(X)
        l = c.labels_
        
        t1 = time.time()
        cost = t1-t0
    
        ari_comp = self.getARI_Comp(X, l)
        # print(batch_index, ari_comp)
        saveStr = str(batch_index)+","+str(ari_comp)+","+str(cost)+"\n"    
        f = open("Output/Rank.csv", 'a')
        f.write(saveStr)
        f.close()
    
    def getARI_Comp(self, X, l):
        # Check if all the labels are -1
        if all(v == -1 for v in l):
            return -1
        n = len(set(l))
        
        if self.determine_param_clustering_algo == "KM":
            c = KMeans(n_clusters=n, n_init=5).fit(X)
        elif self.determine_param_clustering_algo == "DBS":
            c = DBSCAN().fit(X)
        elif self.determine_param_clustering_algo == "HAC":
            c = AgglomerativeClustering(n_clusters = n).fit(X)
            
        a_l = c.labels_
        ari = adjusted_rand_score(l, a_l)
        return ari
            
    
    def rerun(self, batch_count):
        if self.bestParams == []:
            print("Determine best parameters before this step.")
            return
        
        threads = []
        batch_index = 0
        done = False
        while True:
            if done:
                break
            for _ in range(10):
                if batch_index == batch_count:
                    done = True
                    break
                t = threading.Thread(target=self.worker_rerun, args=(self.bestParams,self.X_batches[batch_index], self.y_batches[batch_index], batch_index))
                threads.append(t)
                t.start()
                batch_index += 1
            for t in threads:
                t.join()
        self.mergeClusteringOutputs_AD()
        # self.mergeClusteringOutputs_DistRatio()
        
    def mergeClusteringOutputs_AD(self):
        
        csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
        csv_files.sort()

        while len(csv_files) > 1:
            for i in range(0, len(csv_files)-1, 2):
                file1 = csv_files[i]
                file2 = csv_files[i+1]
                df1 = pd.read_csv(file1)
                df2 = pd.read_csv(file2)
                
                os.remove(file1)
                os.remove(file2)
                
                X_1 = df1.drop(["y", "l"], axis=1).to_numpy()
                labels1 = df1["l"].to_numpy()
                unique_labels1 = set(df1["l"])
                
                X_2 = df2.drop(["y", "l"], axis=1).to_numpy()
                labels2 = df2["l"].to_numpy()
                unique_labels2 = set(df2["l"])
                
                
                if len(unique_labels1) == 1 and len(unique_labels2) == 1:
                    df = pd.concat([df1, df2])
                    df.to_csv(file2, index=False)
                    continue
                #     models=[]
                #     for i in unique_labels1:
                #         v_i = X_1[labels1 == i]
                #         clf = LocalOutlierFactor(n_neighbors=2, novelty=True).fit(v_i)
                #         models.append(clf)
                #     for j in unique_labels2:
                #         v_j = X_2[labels2 == j]
                #         mean_scores = []
                #         for m in models:
                #             lof_predict = m.predict(v_j)
                #             mean_scores.append(np.mean(lof_predict))
                #         print("lof_predict", mean_scores)
                global_centers = []
                global_centers_frequency = []

                global_centers_count = len(unique_labels1)
                for i in unique_labels1:
                    global_centers.append(X_1[labels1 == i].mean(axis=0))
                    global_centers_frequency.append(len(X_1[labels1 == i]))
                
                df2["ll"] = -2
                
                for i in unique_labels2:
                    c = X_2[labels2 == i].mean(axis=0)
                    
                    distances = [np.linalg.norm(np.array(c) - np.array(z)) for z in global_centers]
                    
                    nearest_cluster = distances.index(min(distances))
                    
                    # Check using AD
                    X_train = X_1[labels1 == nearest_cluster]
                    X_test = X_2[labels2 == i]
                    if len(X_train) == 1:
                        new_i2 = global_centers_count
                        global_centers_count += 1
                        df2.loc[df2['l'] == i, 'll'] = new_i2
                        continue
                    ad = LocalOutlierFactor(n_neighbors=2, novelty=True).fit(X_train)
                    predict = ad.predict(X_test)
                    if np.mean(predict) > 0:
                        # TODO: Edit centers
                        new_i2 = nearest_cluster
                    else:
                        new_i2 = global_centers_count
                        global_centers_count += 1
                        # TODO: Add the new cluster info to the global
                    df2.loc[df2['l'] == i, 'll'] = new_i2
                
                df2 = df2.drop("l", axis=1)
                df2 = df2.rename(columns={'ll': 'l'})
                
                df = pd.concat([df1, df2])
                df.to_csv(file2, index=False)
            
            csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
            csv_files.sort()
            
        csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
        df = pd.read_csv(csv_files[0])
        
        df.to_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+".csv", index=False)
        
        # X_ = df.drop(["y", "l"], axis=1).to_numpy()
        # labels = df["l"].to_numpy()
        # unique_labels = set(df["l"])
        # print(unique_labels)
        
        # yy = df["y"].tolist()
        # ll = df["l"].tolist()
        # ari = adjusted_rand_score(yy, ll)
       
        # print("rerun ari: ", ari)
        
        
    def mergeClusteringOutputs_DistRatio(self):
        # Read batch outputs and merge
        global_centers = []
        global_centers_frequency = []
        global_centers_count = 0
        df_csv_append = pd.DataFrame()
        
        csv_files = glob.glob('Output/Temp/*.{}'.format('csv'))
        
        for file in csv_files:
            df = pd.read_csv(file)
            X_c = df.drop(["y", "l"], axis=1).to_numpy()
            labels = df["l"].to_numpy()
            unique_labels = set(df["l"])
            if len(unique_labels) == 1:
                df_csv_append = pd.concat([df_csv_append, df])
                continue
            if global_centers_count == 0:
                global_centers_count = len(unique_labels)
                for i in unique_labels:
                    global_centers.append(X_c[labels == i].mean(axis=0))
                    global_centers_frequency.append(len(X_c[labels == i]))
                df_csv_append = pd.concat([df_csv_append, df])
                continue                        
            
            df["ll"] = -2
        
            for i in unique_labels:
                c = X_c[labels == i].mean(axis=0)

                distances = [np.linalg.norm(np.array(c) - np.array(z)) for z in global_centers]
                sorted_distance = sorted(distances)
                ratio = sorted_distance[0]/sorted_distance[1]
                if ratio < 0.75:
                    
                    new_i = distances.index(sorted_distance[0])
                    
                    number_of_local_points_in_cluster_i = len(X_c[labels == i])
                    number_of_global_points_in_cluster_i = global_centers_frequency[new_i]
                    new_number_of_global_points_in_cluster_i = number_of_local_points_in_cluster_i+number_of_global_points_in_cluster_i

                    global_centers[new_i] = (global_centers[new_i]*number_of_global_points_in_cluster_i + c*number_of_local_points_in_cluster_i)/new_number_of_global_points_in_cluster_i
                    global_centers_frequency[new_i] = new_number_of_global_points_in_cluster_i
                else:
                    new_i = global_centers_count
                    global_centers_count += 1
                    global_centers.append(c)
                    global_centers_frequency.append(len(X_c[labels == i]))
                    
                df.loc[df['l'] == i, 'll'] = new_i                    
            df = df.drop("l", axis=1)
            df = df.rename(columns={'ll': 'l'})
            df_csv_append = pd.concat([df_csv_append, df])
        
        yy = df_csv_append["y"].tolist()
        ll = df_csv_append["l"].tolist()
        ari = adjusted_rand_score(yy, ll)
        print("Total clusters: ", global_centers_count)
        print("rerun ari: ", ari)
        
        # Cluster of clusters
        df_csv_append["lll"] = -2
        if global_centers_count > 100:
            c = AffinityPropagation(damping=0.5).fit(global_centers)
            l = c.labels_
            
            if len(set(l)) > 1:
                for i in range(global_centers_count):
                    df_csv_append.loc[df_csv_append['l'] == i, 'lll'] = l[i]
                df_csv_append = df_csv_append.drop("l", axis=1)
                df_csv_append = df_csv_append.rename(columns={'lll': 'l'})
        
            yy = df_csv_append["y"].tolist()
            ll = df_csv_append["l"].tolist()
            ari = adjusted_rand_score(yy, ll)
            print("Total clusters: ", len(set(l)))
            print("rerun ari: ", ari)
        
        df_csv_append.to_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+".csv", index=False)
        
    def worker_rerun(self, parameter, X, y, batch_index):
        if self.rerun_mode == "A":
            if self.algoName == "AP":
                c = AffinityPropagation(damping=parameter[0], max_iter=parameter[1], convergence_iter=parameter[2]).fit(X)
            elif self.algoName == "SC":
                c = SpectralClustering(n_clusters=self.n_cluster, eigen_solver=parameter[0], n_components=parameter[1], 
                                       n_init=parameter[2], gamma=parameter[3], affinity=parameter[4], 
                                       n_neighbors=parameter[5], assign_labels=parameter[6], 
                                       degree=parameter[7], n_jobs=parameter[8]).fit(X)
            
            
            l = c.labels_
            X["y"] = y
            X["l"] = l
            X.to_csv("Output/Temp/"+str(batch_index)+".csv", index=False)

        if self.rerun_mode == "B":
            ll = []
            for c in self.models:
                ll.append(c.predict(X))

            if self.algoName == "AP":
                c = AffinityPropagation(damping=parameter[0], max_iter=parameter[1], convergence_iter=parameter[2]).fit(X)
            
            elif self.algoName == "SC":
                c = SpectralClustering(n_clusters=self.n_cluster, eigen_solver=parameter[0], n_components=parameter[1], 
                                       n_init=parameter[2], gamma=parameter[3], affinity=parameter[4], 
                                       n_neighbors=parameter[5], assign_labels=parameter[6], 
                                       degree=parameter[7], n_jobs=parameter[8]).fit(X)
                
            l = c.predict(X)
            ll.append(l)
            
            if len(ll) == 1:
                X["y"] = y
                X["l"] = l
                X.to_csv("Output/Temp/"+str(batch_index)+".csv", index=False)
                return
            
            f1Scores = []
            for l_i in range(len(ll)):
                unique_values = set(ll[l_i])
                permutations = list(itertools.permutations(unique_values))
                bestPerm = []
                bestF1 = -1
                for perm in permutations:
                    replacements = {}
                    for i in range(len(unique_values)):
                        replacements[i] = perm[i]
                    new_numbers = self.replace_numbers(ll[l_i], replacements)
                    f1_s = f1_score(l, new_numbers, average='weighted')
                    if f1_s > bestF1:
                        bestF1 = f1_s
                        bestPerm = new_numbers
                f1Scores.append(bestF1)
                ll[l_i] = bestPerm
            
            for i in range(len(ll[0])):
                dict={}
                for j in range(len(ll)):
                    if ll[j][i] in dict:
                        dict[ll[j][i]] += f1Scores[j]
                    else:
                        dict[ll[j][i]] = f1Scores[j]
                
                l[i] = max(dict, key=dict.get)
                
            X["y"] = y
            X["l"] = l
            X.to_csv("Output/Temp/"+str(batch_index)+".csv", index=False) 

            self.models.append(c)
                    
    
    def AUL_ARI(self):
        df = pd.read_csv("ClusteringOutput/"+self.fileName+"_"+self.algoName+".csv")
        
        yy = df["y"].tolist()
        ll = df["l"].tolist()
        ari = adjusted_rand_score(yy, ll)
       
        return ari
    
    def run(self):
        t0 = time.time()
        if self.X.shape[0] < 10000:    
            batch_count = 20
        elif self.X.shape[0] > 10000 and self.X.shape[0] < 100000:    
            batch_count = 100
        else:
            batch_count = int(self.X.shape[0]/100000)*100
        self.subSample(batch_count)
        print("Determine Parameters")
        self.determineParam()
        # batch_count = 100
        self.subSample(batch_count)
        print("Rerun")
        self.rerun(batch_count)
        t1 = time.time()
        ari_ss = self.AUL_ARI()
        time_ss = t1-t0 
        # print("Time: ", time_ss)
        # return 0, 0
        return ari_ss, time_ss
    
    def replace_numbers(numbers, replacements):
        new_numbers = []
        for number in numbers:
            if number in replacements:
                new_numbers.append(replacements[number])
            else:
                new_numbers.append(number)
        return new_numbers
    
    def DetermineOptimalComparisonAlgorithm(self, mode):
        comparison_modes = ["ARI", "ARI_T"]
        comparison_mode_algos = ["KM", "HAC", "DBS"]
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
                str_values=str_values+","+str(f1_ss)+","+str(time_ss)
        
        f=open("Stats/"+self.algoName+"_SubsampleAlgoComp.csv", "a")
        f.write(str_values+'\n')
        f.close()
        print(str_values)
        return f1_ss, time_ss
    
    
def algo_parameters(algo):
    parameters = []
    if algo == "AP":
        damping = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        max_iter = [50, 100, 200, 300, 400, 500, 750, 1000]
        convergence_iter = [5, 10, 15, 20, 25, 30, 40, 50] 
        
        
        parameters.append(["damping", 0.5, damping])
        parameters.append(["max_iter", 200, max_iter])
        parameters.append(["convergence_iter", 15, convergence_iter])
    
    if algo =="SC":
        eigen_solver = ["arpack", "lobpcg"]
        n_components = [None, 2, 3, 4, 5, 6, 10]
        n_init = [1, 5, 10, 15, 20, 30, 50, 100]
        gamma = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        affinity = ["nearest_neighbors", "rbf"]
        n_neighbors = [5, 10, 20, 30, 50, 75, 100]
        assign_labels = ["kmeans", "discretize"]
        degree = [2,3,4,5]
        n_jobs = [None, -1]
        
        
        parameters.append(["eigen_solver", "arpack", eigen_solver])
        parameters.append(["n_components", None, n_components])        
        parameters.append(["n_init", 10, n_init])
        parameters.append(["gamma", 1.0, gamma])   
        parameters.append(["affinity", "rbf", affinity])
        parameters.append(["n_neighbors", 10, n_neighbors])
        parameters.append(["assign_labels", "kmeans", assign_labels])
        parameters.append(["degree", 3, degree])
        parameters.append(["n_jobs", None, n_jobs])  
    
    
    return parameters
 
def ReRunModeTest(algorithm, master_files):
    if os.path.exists("Stats/"+algorithm+"_ModesTest.csv") == 0:
        f=open("Stats/"+algorithm+"_ModesTest.csv", "w")
        f.write('Filename,Shape_R,Shape_C,Size,ARI_A,Time_A,ARI_B,Time_B\n')
        f.close()
        
        
        for file in master_files:
            # try:
            parameters = algo_parameters(algorithm)
        
            algoRun_ss_a = AUL_Clustering(parameters, file, algorithm)
            skip, shape, size = algoRun_ss_a.readData()
            if skip:
                algoRun_ss_a.destroy()
                continue
            print("Start: Sampling with RerunMode 1")
            print(file)
            
            ari_ss_a, time_ss_a = algoRun_ss_a.run()
            print(ari_ss_a, time_ss_a)
            
            algoRun_ss_a.destroy()
            del algoRun_ss_a

            print("Start: Sampling with RerunMode 2")        
            
            algoRun_ss_b = AUL_Clustering(parameters, file, algorithm)
            algoRun_ss_b.rerun_mode = "B"
            skip, shape, size = algoRun_ss_b.readData()
            ari_ss_b, time_ss_b = algoRun_ss_b.run()
            print(ari_ss_b, time_ss_b)
            algoRun_ss_b.destroy()
            del algoRun_ss_b
            
            # # WRITE TO FILE Different Modes
            f=open("Stats/"+algorithm+"_ModesTest.csv", "a")
            f.write(file+','+str(shape[0])+','+str(shape[1])+','+str(size)+','+str(ari_ss_a)+','+str(time_ss_a)+','+str(ari_ss_b)+','+str(time_ss_b)+'\n')
            f.close()
           
if __name__ == '__main__':
    algorithm = "AP"
    
    folderpath = datasetFolderDir
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    if os.path.exists("Stats/"+algorithm+".csv"):
        done_files = pd.read_csv("Stats/"+algorithm+".csv")
        done_files = done_files["Filename"].to_numpy()
        # print(done_files)
        master_files = [x for x in master_files if x not in done_files]
    
    master_files.sort(reverse=True)
    # print(master_files)
    
    # # Test Rerun Modes
    # ReRunModeTest(algorithm, master_files)
    
    
    if os.path.exists("Stats/"+algorithm+".csv") == 0:
        f=open("Stats/"+algorithm+".csv", "w")
        f.write('Filename,Shape_R,Shape_C,Size,ARI,ARI_WD,Time_WD,ARI_SS,Time_SS,ARI_WO,Time_WO\n')
        f.close()
    
    # if os.path.exists("Stats/"+algorithm+"_SubsampleAlgoComp.csv") == 0:
    #     f=open("Stats/"+algorithm+"_SubsampleAlgoComp.csv", "w")
    #     f.write('Filename,F1_F1LOF,Time_F1LOF,F1_F1TLOF,Time_F1TLOF,F1_F1OCSVM,Time_F1OCSVM,F1_F1TOCSVM,Time_F1TOCSVM,F1_F1IF,Time_F1IF,F1_F1TIF,Time_F1TIF,F1_F1EE,Time_F1EE,F1_F1TEE,Time_F1TEE\n')
    #     f.close()
    count = 0
    for file in master_files:
        
        # try:
        parameters = algo_parameters(algorithm)
        
        algoRun_ss = AUL_Clustering(parameters, file, algorithm)
        
        skip, shape, size = algoRun_ss.readData()
        if skip:
            algoRun_ss.destroy()
            continue
        
        print("Start: Sampling with RerunMode 1")
        
        print(file)
        
        ari_ss, time_ss = algoRun_ss.run()
        print(ari_ss, time_ss)
        # # print("Best Parameters: ", algoRun.bestParams)
        algoRun_ss.destroy()
        del algoRun_ss

       
        print("Start: Default without sampling")
        algoRun_ws = AUL_Clustering(parameters, file, algorithm)
        ari, ari_wd, time_wd = algoRun_ws.runWithoutSubsampling("default")
        algoRun_ws.destroy()
        
        # # f1_wo, time_wo = algoRun.runWithoutSubsampling("optimized")
        
        # # WRITE TO FILE
        f=open("Stats/"+algorithm+".csv", "a")
        f.write(file+','+str(shape[0])+','+str(shape[1])+','+str(size)+','+str(ari)+','+str(ari_wd)+','+str(time_wd)+','+str(ari_ss)+','+str(time_ss)+',0,0\n')
        # f.write(file+','+str(ari)+','+str(ari_wd)+','+str(time_wd)+','+str(ari_ss)+','+str(time_ss)+','+str(ari_wo)+','+str(time_wo) +'\n')
        f.close()
            
        # except:
        #     print("Fail")
    
        # break
    

# """
# Shuttil
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 25.1 GiB for an array with shape (58000, 58000) and data type float64

# """