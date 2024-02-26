from PAU.PAU_Clustering import PAU_Clustering
import os
import sys
import glob
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy.stats import gmean

import matplotlib.pyplot as plt

def ModuleWiseTimeDist(algo):
    df = pd.read_csv("Stats/Ablation/Ablation_TimeDist_"+algo+".csv")
    df_bs = pd.read_csv("Stats/Ablation/BatchSizeTestModuleTime_"+algo+".csv")

    df = pd.merge(df, df_bs, on=["Filename", "BatchCount"], how="inner")


    df["TimePart"] = df["TimePart"] / df["Time_x"]
    df["TimeHAPV"] = df["TimeHAPV"] / df["Time_x"]
    df["TimeRerun"] = df["TimeRerun"] / df["Time_x"]
    df["TimeMerge"] = df["TimeMerge"] / df["Time_x"]
    
    
    TimePart = df.groupby('BatchSize')["TimePart"].mean()
    TimeHAPV = df.groupby('BatchSize')["TimeHAPV"].mean()
    TimeRerun = df.groupby('BatchSize')["TimeRerun"].mean()
    TimeMerge = df.groupby('BatchSize')["TimeMerge"].mean()
    
    merged_df = pd.concat([TimePart, TimeHAPV, TimeRerun, TimeMerge], axis=1)
    
    Series = {}
    for ind in merged_df.index:
        Series[f"{ind}"] = merged_df.loc[ind].to_numpy()

    fig, ax = timeDist(Series, ["3.1 Data Partitioning", "3.2 Tuning Parameters for HAPV", "3.3 Generating Labels", "3.4 Merging Labels"])
    fig.savefig("Figures/TimeDist/"+algo+".pdf")


def timeDist(results, category_names):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = ['gold', 'tomato', 'limegreen', 'dodgerblue']


    fig, ax = plt.subplots(figsize=(7, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)
    # ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
    #           loc='lower left', fontsize='small')
    legend = ax.legend(ncol=2, bbox_to_anchor=(0, 1),
                       loc='lower left', fontsize='small', title='')
    # legend.get_title().set_alignment('left')  # Set legend title alignment to left


    return fig, ax

    
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

def RefereeARIvsTime_old():
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

def NoRefTest(algo):
    df_n = pd.read_csv("Stats/Ablation/Ablation_NoRef_"+algo+".csv")
    df_ace = pd.read_csv("Stats/Merged_SS.csv")
    
    df = pd.merge(df_ace, df_n, on='Filename', how='inner')
    df = df.loc[df["ARI"]!=-2]

    df_selected = df[['ARI', 'ARI_'+algo]]
    df_selected = df_selected.rename(columns={'ARI': 'Default', 'ARI_'+algo: 'With Validator'})

    df_melted = df_selected.melt(var_name='Column', value_name='Value')
    
    sns.violinplot(x='Column', y='Value', data=df_melted)
    plt.xlabel('')
    plt.ylabel('ARI')
    plt.title(algo)

    plt.show()

    

def RefereeARIvsTime(algo):
    df = pd.read_csv("Stats/Ablation/Ablation_RefereeClAlgo_"+algo+".csv")
    # sns.boxplot(x='Referee', y='Time', data=df)
    # plt.title(algo + ' - Time')
    # plt.xlabel('Referee')
    # plt.ylabel('Time')
    # plt.show()
    
    # sns.violinplot(x='Referee', y='Time', data=df)
    # plt.title(algo + ' - Time')
    # plt.xlabel('Referee')
    # plt.ylabel('Time')
    # plt.show()

    # sns.boxplot(x='Referee', y='ARI', data=df)
    # plt.title(algo + ' - ARI')
    # plt.xlabel('Referee')
    # plt.ylabel('ARI')
    # plt.show()
    
    # sns.violinplot(x='Referee', y='ARI', data=df)
    # plt.title(algo + ' - ARI')
    # plt.xlabel('Referee')
    # plt.ylabel('ARI')
    # plt.show()
    # Initialize lists to store normalized values
    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    df["ARI"] = scaler.fit_transform(df[["ARI"]])

    
    df["ARI"] = -np.log(df[["ARI"]])
    grouped = df.groupby('Referee')

    df['Normalized_Time'] = df.groupby('Referee')['Time'].transform(lambda x: (x/x.max()))
    df['ARI_Normalized_Time'] = df['ARI'] * df['Normalized_Time']

    sns.violinplot(x='Referee', y='ARI_Normalized_Time', data=df)
    plt.title(algo)
    plt.xlabel('Validator')
    plt.ylabel('$C_{AxT}$', fontsize=14)
    plt.savefig('Figures/Ablation_Ref_'+algo+'.pdf', bbox_inches='tight')
    plt.show()

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
    # df = pd.read_csv("Utility&Test/Stats/BatchSizeTest_"+algo+".csv")
    # df = pd.read_csv("Utility&Test/Stats/BatchSizeTest_"+algo+".csv")
    df = pd.read_csv("Stats/Ablation/BatchSizeTest_"+algo+".csv")

    all_normalized_time = []

    for filename, group in df.groupby('Filename'):
        max_time = group['Time'].max()
        normalized_time = group['Time'] / max_time
        all_normalized_time.append((group['BatchSize'], normalized_time, filename))

        # plt.plot(group['BatchSize'], normalized_time, label=filename)

        # # plt.plot(group['BatchSize'], group['Time'], label=filename)
        
        # plt.title(algo+filename)
        # plt.xlabel('Batch Size')
        # plt.ylabel('Time')
        # # plt.legend()
        # plt.show()
    
    for batch_sizes, normalized_times, filename in all_normalized_time:
        plt.plot(batch_sizes, normalized_times, label=filename)
    plt.title(algo + " - Time")
    plt.xlabel('Batch Size')
    plt.ylabel('Normalized Time')
    # plt.legend()
    plt.show()

    all_normalized_ari = []
    for filename, group in df.groupby('Filename'):
        all_normalized_ari.append((group['BatchSize'], group['ARI'], filename))

        # plt.plot(group['BatchSize'], group['ARI'], label=filename)
        
        # plt.title(filename)
        # plt.xlabel('Batch Size')
        # plt.ylabel('ARI')
        # # plt.legend()
        # plt.show()
    
    for batch_sizes, ari, filename in all_normalized_ari:
        plt.plot(batch_sizes, ari, label=filename)
    plt.title(algo + " - ARI")
    plt.xlabel('Batch Size')
    plt.ylabel('ARI')
    # plt.legend()
    plt.show()


def BatchAvgPlot(algo, Y, color):
    df = pd.read_csv("Stats/Ablation/BatchSizeTest_"+algo+".csv")

    grouppedByFilename = []

    for filename, group in df.groupby('Filename'):
        if Y == "Time":
            max_time = group['Time'].max()
            group_y = group['Time'] / max_time
        else:
            group_y = group['ARI']
        grouppedByFilename.append((group['BatchSize'], group_y, filename))
        
    batch_dict = {}
    for batch_sizes, y, filename in grouppedByFilename:
        for i in batch_sizes.index:
            if batch_sizes[i] not in batch_dict:
                batch_dict[batch_sizes[i]] = []
            batch_dict[batch_sizes[i]].append(y[i])
    ys = []
    xs = []
    for k, l in batch_dict.items():
        avg = np.mean(l)
        ys.append(avg)
        xs.append(k)
    drawPolyFit(xs, ys, algo, 'Partition Size', Y, color)
    
def drawPolyFit(x, y, algo, x_label, y_label, color):
    if algo == "SC":
        algo = "SpecC"
    degree = 2
    coefficients = np.polyfit(x, y, degree)
    poly_function = np.poly1d(coefficients)
    # x_fit = np.linspace(min(x), 2500, 100)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly_function(x_fit)
    plt.plot(x, y, "o", label=f'{algo}', color=color)
    plt.plot(x_fit, y_fit, '--', label='', color=color)

    # plt.title(algo)
    plt.xlabel(x_label)
    plt.ylabel((y_label if y_label is not "Time" else "Normalized Time"))
    plt.legend()
    # plt.show() # Remove this comment for separate plots

def BatchTest():
    BatchAvgPlot("DBSCAN", "Time", "#FFA500")
    BatchAvgPlot("AP", "Time", "#6F00FF")
    BatchAvgPlot("HAC", "Time", "#32CD32")
    BatchAvgPlot("SC", "Time", "#FF00FF")
    plt.savefig('Figures/Ablation_Batch_Time.pdf', bbox_inches='tight')
    plt.show()
    BatchAvgPlot("DBSCAN", "ARI", "#FFA500")
    BatchAvgPlot("AP", "ARI", "#6F00FF")
    BatchAvgPlot("HAC", "ARI", "#32CD32")
    BatchAvgPlot("SC", "ARI", "#FF00FF")
    plt.savefig('Figures/Ablation_Batch_ARI.pdf', bbox_inches='tight')
    plt.show()
    
    

# BatchTest()
# BoxPlotReferee()
# BoxPlotMode()
# Batch("DBSCAN")

# # Batch("SC")
# RefereeARIvsTime("HAC")
# RefereeARIvsTime("AP")

# ScatterReferee()

NoRefTest("AP")
NoRefTest("HAC")
NoRefTest("DBSCAN")
# NoRefTest("SC")

ModuleWiseTimeDist("AP")
ModuleWiseTimeDist("DBSCAN")
ModuleWiseTimeDist("HAC")
ModuleWiseTimeDist("SC")
