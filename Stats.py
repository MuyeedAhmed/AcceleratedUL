import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import ttest_ind






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
        
        
    print(len(ss), len(default))
    t_statistic, p_value = ttest_ind(ss, default)
    
    print(t_statistic, p_value)


# df_default = pd.read_csv("Stats/HAC/Louise.csv")
# df_ss = pd.read_csv("Stats/HAC/Thelma.csv")

# # plot_time(df_default, df_ss)
# # calculate_ari_diff(df_default, df_ss)
# ttest(df_default, df_ss)

def MergeSS():
    df_AP = pd.read_csv("Stats/AP/M2.csv")
    df_DBSCAN = pd.read_csv("Stats/DBSCAN/3070.csv")
    df_HAC = pd.read_csv("Stats/HAC/M2.csv")
    df_SC = pd.read_csv("Stats/SC/M2.csv")
    
    df_AP.drop_duplicates(subset='Filename', keep='first', inplace=True)
    df_DBSCAN.drop_duplicates(subset='Filename', keep='first', inplace=True)
    df_HAC.drop_duplicates(subset='Filename', keep='first', inplace=True)
    df_SC.drop_duplicates(subset='Filename', keep='first', inplace=True)

    
    merged_df1 = pd.merge(df_AP, df_DBSCAN, on=["Filename","Row","Columm","Mode"], suffixes=('_AP', '_DBSCAN'))
    merged_df2 = pd.merge(df_HAC, df_SC, on=["Filename","Row","Columm","Mode"], suffixes=('_HAC', '_SC'))
    
    merged_df = pd.merge(merged_df1, merged_df2, on=["Filename","Row","Columm","Mode"])
    merged_df.to_csv("Stats/Merged_SS.csv", index=False)
    # print(merged_df)
    
# MergeSS()

def MergeDefault_OuterJoin():
    df_AP = pd.read_csv("Stats/AP/Jimmy.csv")
    df_DBSCAN = pd.read_csv("Stats/DBSCAN/M2.csv")
    df_HAC = pd.read_csv("Stats/HAC/3070.csv")
    df_SC = pd.read_csv("Stats/SC/Jimmy.csv")
    
    
    df_AP.drop_duplicates(subset='Filename', keep='first', inplace=True)
    df_DBSCAN.drop_duplicates(subset='Filename', keep='first', inplace=True)
    df_HAC.drop_duplicates(subset='Filename', keep='first', inplace=True)
    df_SC.drop_duplicates(subset='Filename', keep='first', inplace=True)
    
    merged_df1 = pd.merge(df_AP, df_DBSCAN, on=["Filename","Row","Columm","Mode"],how='outer', suffixes=('_AP', '_DBSCAN'))
    merged_df2 = pd.merge(df_HAC, df_SC, on=["Filename","Row","Columm","Mode"], how='outer', suffixes=('_HAC', '_SC'))
    
    merged_df = pd.merge(merged_df1, merged_df2, on=["Filename","Row","Columm","Mode"], how='outer')
    
    merged_df.to_csv("Stats/Merged_Default.csv", index=False)
    

    
    df_SS = pd.read_csv("Stats/Merged_SS.csv")
    ss_files = df_SS["Filename"].to_numpy()
    
    filtered_df = merged_df[merged_df['Filename'].isin(ss_files)]
    filtered_df.to_csv("Stats/Merged_Default_Filtered.csv", index=False)

    
# MergeSS()
# MergeDefault_OuterJoin()

def boxPlot(algo):
    df_SS = pd.read_csv("Stats/Merged_SS.csv")
    df_Default = pd.read_csv("Stats/Merged_Default_Filtered.csv")

    ss = df_SS["ARI_"+algo].to_numpy()
    default = df_Default["ARI_"+algo].to_numpy()
    default = [x for x in default if str(x) != 'nan']
    
    # print(default)
    my_dict = {'AUL': ss, 'Default': default}

    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    
# boxPlot("SC")



def CalculateAvg(Algo):
    df_SS = pd.read_csv("Stats/Merged_SS.csv")
    df_algo_ss = pd.read_csv("Stats/"+Algo+"/SS.csv")
    df_default = pd.read_csv("Stats/Merged_Default_Filtered.csv")
    df_algo_default = pd.read_csv("Stats/"+Algo+"/Default.csv")
    
    df_algo_ss.drop_duplicates(subset='Filename', keep='first', inplace=True)
    df_algo_default.drop_duplicates(subset='Filename', keep='first', inplace=True)
    
    merged_df_SS = pd.merge(df_algo_ss, df_SS, on=["Filename","Row","Columm","Mode"], suffixes=('_DF', '_SS'))
    
    merged_df_default = pd.merge(df_algo_default, df_default, on=["Filename","Row","Columm","Mode"], suffixes=('_DF', '_Default'))
    
    
    # print("merged_df_default count", merged_df_default.count())
    # print("df_algo_default count", df_algo_default.count())
    # print("df_default count", df_default.count())

    print("SS (ALL)")
    print("M2: ", merged_df_SS["Time_M2"].mean())
    print("Jimmy: ", merged_df_SS["Time_Jimmy"].mean())
    print("Thelma: ", merged_df_SS["Time_Thelma"].mean())
    print("Louise: ", merged_df_SS["Time_Louise"].mean())
    
    
    print("Default (R)")
    print("M2: ", merged_df_default["Time_M2"].mean())
    print("Jimmy: ", merged_df_default["Time_Jimmy"].mean())
    print("Thelma: ", merged_df_default[merged_df_default["Time_Thelma"] <= 7200]["Time_Thelma"].mean(), merged_df_default[merged_df_default["Time_Thelma"] > 7200]["Time_Thelma"].count())
    print("Louise: ", merged_df_default[merged_df_default["Time_Louise"] <= 7200]["Time_Louise"].mean(), merged_df_default[merged_df_default["Time_Louise"] > 7200]["Time_Louise"].count())
    
    default_names = merged_df_default["Filename"]
    SS_R = pd.merge(merged_df_SS, default_names, on=["Filename"])
    print("SS (R)")
    print("M2: ", SS_R["Time_M2"].mean())
    print("Jimmy: ", SS_R["Time_Jimmy"].mean())
    
    
    print("Thelma: ", SS_R[SS_R["Filename"].isin(merged_df_default[merged_df_default["Time_Thelma"] <= 7200]["Filename"].to_numpy())]["Time_Thelma"].mean())
    print("Louise: ", SS_R[SS_R["Filename"].isin(merged_df_default[merged_df_default["Time_Louise"] <= 7200]["Filename"].to_numpy())]["Time_Louise"].mean())

    # print("Louise: ", SS_R["Time_Louise"].mean())
    
    
CalculateAvg("DBSCAN")



# def hacfix():
#     df_1 = pd.read_csv("Stats/HAC/3070.csv")
#     df_2 = pd.read_csv("Stats/HAC/Louise.csv")
#     # df_3 = pd.read_csv("Stats/HAC/Louise copy.csv")
    
#     df_1.drop_duplicates(subset='Filename', keep='first', inplace=True)
#     df_2.drop_duplicates(subset='Filename', keep='first', inplace=True)
#     # df_3.drop_duplicates(subset='Filename', keep='first', inplace=True)
    
#     # print(df_3.shape)
    
#     df_4 = pd.merge(df_1, df_2, on=["Filename","Row","Columm","Mode"], how='outer', suffixes=('_3070', '_Louise'))
#     df_4.to_csv("Stats/HAC/Default.csv", index=False)
    
#     # df_5 = pd.merge(df_4, df_3, on=["Filename","Row","Columm","Mode"], how='outer', suffixes=('_o', '_c'))
#     # df_5.to_csv("Stats/HAC/Default copy.csv", index=False)
    
# hacfix()
    
def MemoryConsumptionCalculation(algo, mode, system):

    memory = pd.read_csv("MemoryStats/Memory_" + algo + "_" + mode + "_" + system + ".csv")
    time = pd.read_csv("MemoryStats/Time_" + algo + "_" + mode + "_" + system + ".csv") 
    
    
    
    
    time = time[time["Completed"] == 1]
    
    time["TotalTime"] = time["EndTime"] - time["StartTime"]
    time['TotalTime'] = time['TotalTime'].mask(time['TotalTime'] < 0, np.nan)

    time["Memory_Max"] = None
    
    
    for index, row in time.iterrows():
        t = memory[(memory["Time"] > row["StartTime"]) & (memory["Time"] < row["EndTime"])]
        if t.empty:
            print(row["Filename"], " is empty!")
            continue
        
        memory_virtual = t["Memory_Virtual"].to_numpy()
        mv_max = np.max(memory_virtual)
        
        time.loc[index, "Memory_Max"] = int(mv_max)
    # print(time)
    
    """"""
    lrd = pd.read_csv("Stats/DBSCAN/M2_lrd.csv")
    label_stats = pd.read_csv("Stats/DBSCAN/M2_Uniq&Outlier.csv") 
    """"""
    
    
    dbscan = time.join(label_stats.set_index('Filename'), lsuffix='_caller', rsuffix='_other', on='Filename')
    
    print(lrd)
    print(dbscan)
    
    dbscan = dbscan.set_index('Filename').join(lrd.set_index('Filename'), lsuffix='_caller', rsuffix='_other')
    
    # dbscan = pd.concat([time, label_stats], axis=1, join="inner", ignore_index=True)
    
    
    dbscan.to_csv("Stats/DBSCAN/With Memory.csv")
    
    # table = time.pivot(index='Row', columns='Columm', values='Memory_Max')
    # table.to_csv("Max_Memory_Usage_" + algo + "_" + mode + "_" + system + ".csv")

# MemoryConsumptionCalculation("DBSCAN", "Default", "M2")

