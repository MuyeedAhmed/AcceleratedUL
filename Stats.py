import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import ttest_ind

import seaborn as sns



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
    df_Default = df_Default.dropna(subset=["ARI_"+algo])
    
    
    df_SS_filtered = df_SS.copy()
    for index, row in df_SS_filtered.iterrows():
        if row['Filename'] not in df_Default['Filename'].values:
            df_SS_filtered.drop(index, inplace=True)
    
    
    ss = df_SS["ARI_"+algo].to_numpy()
    default = df_Default["ARI_"+algo].to_numpy()
    ss_filtered = df_SS_filtered["ARI_"+algo].to_numpy()

    default = [x for x in default if str(x) != 'nan']
    ss_filtered = [x for x in ss_filtered if str(x) != 'nan']
    
    print("Mean, Median")
    print(np.mean(default), np.median(default))
    print(np.mean(ss_filtered), np.median(ss_filtered))
    print(np.mean(ss), np.median(ss))

    longer_list = ss if len(ss) > len(default) else default
    shorter_list = ss if len(ss) <= len(default) else default
    

    num_missing = len(longer_list) - len(shorter_list)
    shorter_list += [np.nan] * num_missing
    
    ss_filtered += [np.nan] * num_missing
    
    if algo == "SC":
        df_001 = pd.read_csv("Stats/Time/SC/Jimmy_0.001_ARI.csv")
        def_001 = df_001["ARI"].to_numpy()
        def_001 = [x for x in def_001 if str(x) != 'nan']
        num_missing_def_001 = len(longer_list) - len(def_001)
        def_001 += [np.nan] * num_missing_def_001
        
        my_dict = {'Default (R)': default, 'Tol:0.001': def_001, 'SAC (R)':ss_filtered, 'SAC (All Datasets)': ss}
        
        
    else:
        my_dict = {'Default (R)': default, 'SAC (R)':ss_filtered, 'SAC (All Datasets)': ss}
    
    
    df = pd.DataFrame(my_dict)
    
    if algo == "SC":
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = fig.add_axes([0.21, 0.4, 0.4, 0.4])
        data_subset = df[['Default (R)', 'Tol:0.001', 'SAC (R)']]

        
        sns.boxplot(data=df, ax=ax1)
        sns.boxplot(data=data_subset, ax=ax2)

        ax1.set_ylabel('ARI', fontsize=12)
        ax1.tick_params(axis='both', labelsize=12)
        ax2.tick_params(axis='both', labelsize=12)

    else:        
        fig, ax = plt.subplots(figsize=(8, 6))
        # ax.boxplot(my_dict.values())
        # ax.set_xticklabels(my_dict.keys())
        
        sns.boxplot(data=df)
        plt.ylabel('ARI')
        # plt.yscale('log')
    fig.savefig('Figures/ARI_'+algo+'.pdf', bbox_inches='tight')
    
boxPlot("SC")



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
    # print("M2: ", merged_df_default["Time_M2"].mean())
    # print("Jimmy: ", merged_df_default["Time_Jimmy"].mean())
    print("M2: ", merged_df_default[merged_df_default["Time_M2"] <= 7200]["Time_M2"].mean(), merged_df_default[merged_df_default["Time_M2"] > 7200]["Time_M2"].count())
    print("Jimmy: ", merged_df_default[merged_df_default["Time_Jimmy"] <= 7200]["Time_Jimmy"].mean(), merged_df_default[merged_df_default["Time_Jimmy"] > 7200]["Time_Jimmy"].count())
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
    
    
# CalculateAvg("SC")



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

def EstimatedTimeFilter():
    df_est = pd.read_csv("Stats/Time/AP/M2.csv")
    df_dataset_list = pd.read_csv("MemoryStats/FileList.csv")
    
    data_list = df_dataset_list["Filename"].to_numpy()
    
    df_est = df_est[df_est['Filename'].isin(data_list)]
    
    f = df_est[df_est["Row"] < 84000]
    
    print(f["Estimated_Time"])
# EstimatedTimeFilter()
    

def ari_stats(algo):
    df_SS = pd.read_csv("Stats/Merged_SS.csv")
    df_Default = pd.read_csv("Stats/Merged_Default_Filtered.csv")
    
    ss = df_SS[["Filename", "ARI_"+algo]]
    default = df_Default[["Filename", "ARI_"+algo]]
    
    ss.rename(columns={"ARI_"+algo: 'ss'}, inplace=True)
    default.rename(columns={"ARI_"+algo: 'default'}, inplace=True)
    
    ss = ss.dropna()
    default = default.dropna()
    
    merged_df = ss.merge(default, on='Filename')

    merged_df['Winner'] = 'None'  # Initialize with 'None'
    merged_df.loc[merged_df['ss'] > merged_df['default'], 'Winner'] = 'ss'
    merged_df.loc[merged_df['default'] > merged_df['ss'], 'Winner'] = 'default'

    win_counts = merged_df['Winner'].value_counts()

    
    print(win_counts)

ari_stats("SC")