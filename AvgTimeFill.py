import pandas as pd
import time
from SS_Clustering import SS_Clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture




def AvgTime(algo, mode):
    folderpath = '../Openml/'    
    
    print(algo, mode)

    files = ["har_OpenML", "pendigits_OpenML", "eye_movements_OpenML"] 
    for filename in files:
        df = pd.read_csv(folderpath+filename+".csv")
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)
        
        
        t0 = time.time()
        if mode == "Default":
    
            if algo == "DBSCAN":
                clustering = DBSCAN(algorithm="brute").fit(X)
                l = clustering.labels_
            elif algo == "AP":
                clustering = AffinityPropagation().fit(X)
                l = clustering.labels_
            elif algo == "GMM":
                clustering = GaussianMixture(n_components=2).fit(X)
                l = clustering.predict(X)
            elif algo == "SC":
                clustering = SpectralClustering().fit(X)
                l = clustering.labels_
            elif algo == "HAC":
                clustering = AgglomerativeClustering().fit(X)
                l = clustering.labels_
           
            time_ = time.time()-t0    
            
        else:
            clustering = SS_Clustering(algoName=algo)
            clustering.X = X
            clustering.y = y
            ari, time_ = clustering.run()
            clustering.destroy()
        
        print("\n\n****\n\n")
        print("***", filename, time_)
        
AvgTime("AP", "SS")
AvgTime("AP", "Default")
