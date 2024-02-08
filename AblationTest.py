from PAU.PAU_Clustering import PAU_Clustering
import os
import sys
import glob
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

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

def RefereeARIvsTime():
    df = pd.read_csv("Utility&Test/Stats/DBSCAN_Ablation_NoAnomaly.csv")

    comp_algos = ["AP", "KM", "DBS", "HAC", "AVG"] 
    for comp_algo in comp_algos:
        cols_with_ari_ = [col for col in df.columns if 'ARI_'+comp_algo in col]
        df['ARI_'+comp_algo] = df[cols_with_ari_].mean(axis=1)
        cols_with_time = [col for col in df.columns if 'Time_'+comp_algo in col]
        df['Time_'+comp_algo] = df[cols_with_time].mean(axis=1)
    
    ARIs = ['ARI_AP', 'ARI_KM', 'ARI_DBS',  'ARI_HAC', 'ARI_AVG']
    Times = ['Time_AP', 'Time_KM', 'Time_DBS', 'Time_HAC', 'Time_AVG']
    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    df[ARIs] = scaler.fit_transform(df[ARIs])

    
    df[ARIs] = -np.log(df[ARIs])
    
    multiplied_columns = []
    for ari_col, time_col in zip(ARIs, Times):
        multiplied_column = df[ari_col] * df[time_col]
        multiplied_column.name = ari_col.split('_')[1]
        multiplied_columns.append(multiplied_column)
    df = pd.concat(multiplied_columns, axis=1)
    
    print(df)
    
    sns.boxplot(df)
    plt.title('DBSCAN')
    plt.xlabel('Algorithms')
    plt.ylabel('ARI x Time')
    plt.show()
    
    sns.violinplot(data=df, inner="quartile", linewidth=1.5)
    plt.title('DBSCAN')
    plt.xlabel('Algorithms')
    plt.ylabel('ARI x Time')
    plt.show()
    
RefereeARIvsTime()

def ScatterReferee():
    df = pd.read_csv("Utility&Test/Stats/DBSCAN_Ablation_NoAnomaly.csv")
    # df = pd.read_csv("Utility&Test/Stats/AP_Ablation.csv")
    comp_algos = ["AP", "KM", "DBS", "HAC", "AVG"] 
    for comp_algo in comp_algos:
        cols_with_ari_ = [col for col in df.columns if 'ARI_'+comp_algo in col]
        df['ARI_'+comp_algo] = df[cols_with_ari_].mean(axis=1)
        cols_with_time = [col for col in df.columns if 'Time_'+comp_algo in col]
        df['Time_'+comp_algo] = df[cols_with_time].mean(axis=1)
    
    
    # comp_algos = ["AP", "KM", "DBS", "HAC", "AVG"] 

    size = 15
    plt.scatter(df['ARI_AP'], df["Time_AP"], label="AP", color='red', s=size)
    plt.scatter(df['ARI_KM'], df["Time_KM"], label="KM", color='blue', s=size)
    plt.scatter(df['ARI_DBS'], df["Time_DBS"], label="DBS", color='green', s=size)
    plt.scatter(df['ARI_HAC'], df["Time_HAC"], label="HAC", color='orange', s=size)
    # plt.scatter(df['ARI_AVG'], df["Time_AVG"], label="Avg", color='purple', s=size)
    
    plt.title('db')
    plt.xlabel('ARI')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.show()

def Batch(algo):
    df = pd.read_csv("Utility&Test/Stats/BatchSizeTest_"+algo+".csv")
    # df = pd.read_csv("Utility&Test/Stats/BatchSizeTest_"+algo+".csv")

    for filename, group in df.groupby('Filename'):
        # if "vote" not in filename:
        #     continue
        plt.plot(group['BatchSize'], group['Time'], label=filename)
        
        plt.title(algo+filename)
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
# Batch("HAC")

# ScatterReferee()