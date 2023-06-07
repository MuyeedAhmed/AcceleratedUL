import os
import shutil
import glob
import pandas as pd
import numpy as np
import random
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
from scipy.spatial.distance import pdist, squareform

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from pathlib import Path

datasetFolderDir = '/Users/muyeedahmed/Desktop/Research/Dataset/'


def readData(filename):
    df = pd.read_csv(datasetFolderDir+filename+".csv")
    row = df.shape[0]
    col = df.shape[1]
    
    if "target" in df.columns:
        y=df["target"].to_numpy()
        X=df.drop("target", axis=1)
    elif "class" in df.columns:
        y=df["class"].to_numpy()
        X=df.drop("class", axis=1)

    u = len(set(y))
    
    f=open("Stats/FileStats.csv", "a")
    f.write(filename+","+str(row)+","+str(col)+","+str(u)+"\n")
    f.close()

    
if __name__ == '__main__':
    algorithm = "DBSCAN"
    
    folderpath = datasetFolderDir
    master_files = glob.glob(folderpath+"*.csv")
    
    for i in range(len(master_files)):
        # master_files[i] = master_files[i].split("/")[-1].split(".")[0]
        master_files[i] = Path(master_files[i]).stem
    master_files.sort()

    
    
    
    if os.path.exists("Stats/FileStats.csv") == 0:
        f=open("Stats/FileStats.csv", "w")
        f.write('Filename,Shape_R,Shape_C,Unique\n')
        f.close()
    
    remaining = len(master_files)
    for file in master_files:
        
        readData(file)
        
        print(remaining)
        remaining-=1