import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import ttest_ind


df_default = pd.read_csv("Stats/HAC/Louise.csv")
df_ss = pd.read_csv("Stats/HAC/Thelma.csv")



def plot_time(df_default, df_ss):
    diff = []
    x = []
    y = []
    t = []
    for index, row in df_default.iterrows():
        ss_ari = df_ss[df_ss["Filename"] == row["Filename"]]
        if ss_ari.empty:
            print(row["Filename"], " not there in SS")
            continue
        x.append(row["Row"])
        y.append(row["Columm"])
        t.append(row["Time"])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.view_init(elev=60, azim=270, roll=0)
    ax.scatter(x, y, t)
        
def calculate_ari_diff(df_default, df_ss):
    diff = []
    row = []
    for index, row in df_default.iterrows():
        ss_ari = df_ss[df_ss["Filename"] == row["Filename"]]
        if ss_ari.empty:
            print(row["Filename"], " not there in SS")
            continue
        diff.append(ss_ari["ARI"].to_numpy()[0] - row['ARI'])
        row.append(row["Row"])
    #plot with row
    plt.figure()
    plt.plot(row, diff, ".")
    #plot sorted
    plt.figure()
    diff.sort()
    plt.plot(diff)
    
## P-value
def ttest(df_default, df_ss):
    default = []
    ss = []
    for index, row in df_default.iterrows():
        ss_ari = df_ss[df_ss["Filename"] == row["Filename"]]
        if ss_ari.empty:
            print(row["Filename"], " not there in SS")
            continue
        default.append(row["ARI"])
        ss.append(ss_ari["ARI"].to_numpy()[0])
        
        
    
    t_statistic, p_value = ttest_ind(ss, default)
    
    print(t_statistic, p_value)



plot_time(df_default, df_ss)
calculate_ari_diff(df_default, df_ss)
ttest(df_default, df_ss)


