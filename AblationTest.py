from PAU.PAU_Clustering import PAU_Clustering
import os
import sys
import glob
import pandas as pd
import time
import numpy as np

import matplotlib.pyplot as plt

def BoxPlotMode():
    df_A = pd.read_csv("Utility&Test/Stats/AP_Best_Subsample_Run_ModeA.csv")
    df_B = pd.read_csv("Utility&Test/Stats/AP_Best_Subsample_Run_ModeB.csv")
    
    merged_df = pd.merge(df_A, df_B, on='Filename', suffixes=('_A', '_B'))

    plt.boxplot([merged_df['ARI_Mean_A'], merged_df['ARI_Mean_B']], labels=['A', 'B'])
    plt.title('AP')
    plt.xlabel('Modes')
    plt.ylabel('ARI')
    plt.show()

    plt.boxplot([merged_df['Time_A'], merged_df['Time_B']], labels=['A', 'B'])
    plt.title('AP')
    plt.xlabel('Modes')
    plt.ylabel('Time')
    plt.show()

    
def BoxPlotReferee():
    df = pd.read_csv("Utility&Test/Stats/DBSCAN_Ablation_NoAnomaly.csv")
    
    comp_algos = ["AP", "KM", "DBS", "HAC", "AVG"] 
    for comp_algo in comp_algos:
        cols_with_ari_ = [col for col in df.columns if 'ARI_'+comp_algo in col]
        df['ARI_'+comp_algo] = df[cols_with_ari_].mean(axis=1)
        cols_with_time = [col for col in df.columns if 'Time_'+comp_algo in col]
        df['Time_'+comp_algo] = df[cols_with_time].mean(axis=1)
    
    ARIs = ['ARI_AP', 'ARI_KM', 'ARI_DBS',  'ARI_HAC', 'ARI_AVG']
    Times = ['Time_AP', 'Time_KM', 'Time_DBS', 'Time_HAC', 'Time_AVG']
    
    plt.boxplot(df[ARIs].values, labels=ARIs)
    plt.title('DBSCAN')
    plt.xlabel('Algorithms')
    plt.ylabel('ARI')
    plt.show()

    plt.boxplot(df[Times].values, labels=Times)
    plt.title('DBSCAN')
    plt.xlabel('Algorithms')
    plt.ylabel('Time(s)')
    plt.show()

def Batch():
    df = pd.read_csv("Utility&Test/Stats/BatchSizeTest_AP.csv")
    # df = pd.read_csv("Utility&Test/Stats/BatchSizeTest_HAC.csv")

    for filename, group in df.groupby('Filename'):
        # if "vote" not in filename:
        #     continue
        plt.plot(group['BatchSize'], group['Time'], label=filename)
        
        plt.title(filename)
        plt.xlabel('Batch Size')
        plt.ylabel('Time')
        # plt.legend()
        plt.show()
    
    for filename, group in df.groupby('Filename'):
        # if "vote" not in filename:
        #     continue
        plt.plot(group['BatchSize'], group['ARI'], label=filename)
        
        plt.title(filename)
        plt.xlabel('Batch Size')
        plt.ylabel('ARI')
        # plt.legend()
        plt.show()


# BoxPlotReferee()
# BoxPlotMode()
Batch()