import pandas as pd
import time
from PAU.PAU_Clustering import PAU_Clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture




def AvgTime(algo, mode):
    folderpath = '../Openml/'    
    
    print(algo, mode)

    # files = ["har_OpenML", "pendigits_OpenML", "eye_movements_OpenML"]
    # files = ["letter_OpenML", "BNG(autos_COMMA_nominal_COMMA_1000000)_OpenML", "BNG(lymph_COMMA_nominal_COMMA_1000000)_OpenML"] 
    # files = ["numerai28.6_OpenML", "BNG(vote)_OpenML", "RandomRBF_50_1E-4_OpenML"] # DBSCAN
    files = ["eye_movements_OpenML", "gas-drift-different-concentrations_OpenML", "PhishingWebsites_OpenML", "2dplanes_OpenML"] # SC
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
            clustering = PAU_Clustering(algoName=algo)
            clustering.X = X
            clustering.y = y
            ari, time_ = clustering.run()
            clustering.destroy()
        
        print("***", filename, time_)
        
# AvgTime("AP", "SS")
# AvgTime("AP", "Default")


# AvgTime("DBSCAN", "SS")
# AvgTime("DBSCAN", "Default")


AvgTime("HAC", "SS")
AvgTime("HAC", "Default")

# AvgTime("SC", "SS")
# AvgTime("SC", "Default")


