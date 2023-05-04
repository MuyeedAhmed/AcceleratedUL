import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
import numpy as np

# df1 = pd.read_csv("Stats/AP.csv")
# df2 = pd.read_csv("Stats/AP_Ablation.csv")

df1 = pd.read_csv("Stats/GMM.csv")
df2 = pd.read_csv("Stats/GMM_Ablation.csv")



df = df1.merge(df2, on='Filename', how='inner')

ARI_Default = df["ARI_WD"]
Time_Default = df["Time_WD"]

print(f"Default:\n\tTime:{np.mean(Time_Default.to_numpy())}\n\tARI:{np.mean(ARI_Default.to_numpy())}")

DeterParamComp = ["KM", "DBS", "HAC", "INERTIA", "AVG"]
RerunModes = ["A", "B"]
MergeModes = ["Distance", "DistanceRatio", "ADLOF", "ADIF", "ADEE", "ADOCSVM"]
# MergeADAlgo = ["LOF", "IF", "EE", "OCSVM"]

stats = pd.DataFrame(columns=['DeterParamComp', 'RerunModes', 'MergeModes', 'Time', 'ARI'])

maxTime = 0
maxTimeColumn = ""
maxARI = -1
maxARIColumn = ""
Time = []
ARI = []




for dpc in DeterParamComp:
    for rm in RerunModes:
        for mm in MergeModes:
            # if mm == "AD":
                # for maa in MergeADAlgo:
                #     columnNameT = "Time_"+dpc+"_"+rm+"_"+mm+maa
                #     columnNameA = "ARI_"+dpc+"_"+rm+"_"+mm+maa
                #     t = np.mean(df[columnNameT].to_numpy())
                #     a = np.mean(df[columnNameA].to_numpy())
                #     Time.append(t)
                #     ARI.append(a)

                #     stats.loc[-1] = [dpc, rm, mm, maa, t, a]
                #     stats.index = stats.index + 1  
                #     if t > maxTime:
                #         maxTime = t
                #         maxTimeColumn = columnNameT
                #     if a > maxARI:
                #         maxARI = a
                #         maxARIColumn = columnNameA
            # else:
            columnNameT = "Time_"+dpc+"_"+rm+"_"+mm
            columnNameA = "ARI_"+dpc+"_"+rm+"_"+mm
            t = np.mean(df[columnNameT].to_numpy())
            a = np.mean(df[columnNameA].to_numpy())
            Time.append(t)
            ARI.append(a)

            stats.loc[-1] = [dpc, rm, mm, t, a]
            stats.index = stats.index + 1  
            if t > maxTime:
                maxTime = t
                maxTimeColumn = columnNameT
            if a > maxARI:
                maxARI = a
                maxARIColumn = columnNameA

#DeterParamComp
groups = stats.groupby('DeterParamComp')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.ARI, group.Time, marker='o', linestyle='', ms=5, label=name)
ax.legend()
plt.xlabel("ARI")
plt.ylabel("Time")
plt.show()


#RerunModes
groups = stats.groupby('RerunModes')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.ARI, group.Time, marker='o', linestyle='', ms=5, label=name)
ax.legend()
plt.xlabel("ARI")
plt.ylabel("Time")
plt.show()

#MergeModes
groups = stats.groupby('MergeModes')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.ARI, group.Time, marker='o', linestyle='', ms=5, label=name)
ax.legend()
plt.xlabel("ARI")
plt.ylabel("Time")
plt.show()


plt.scatter(ARI, Time)
plt.xlabel("ARI")
plt.ylabel("Time")

print(maxARI)
print(maxARIColumn)
# SS = df["ARI_SS"]

# Optimized = df["ARI_WO"]

# df["Difference"] = Default - SS

# df = df.sort_values('Difference')

# diff = df["Difference"].to_numpy()

# plt.figure(0)

# plt.plot(diff)

# plt.figure(1)
# SS.plot.kde()
# Default.plot.kde()

# plt.figure(2)
# SS.reset_index().plot.scatter(x='index', y='ARI_SS', color='red')
# Default.reset_index().plot.scatter(x='index', y='ARI_WD', color='blue')



# # stat, p = ttest_ind(group1, group2)

# _, o_d_t = ttest_ind(Optimized, Default)
# _, o_s_t = ttest_ind(Optimized, SS)
# _, s_d_t = ttest_ind(SS, Default)


# # _, o_d_t = mannwhitneyu(x=Optimized, y=Default, alternative = 'two-sided')
# # _, s_d_t = mannwhitneyu(x=SS, y=Default, alternative = 'two-sided')


# # _, o_d_g = mannwhitneyu(x=Optimized, y=Default, alternative = 'greater', use_continuity=True)
# # _, s_d_g = mannwhitneyu(x=SS, y=Default, alternative = 'greater', use_continuity=True)


# print('Default vs Optimal: p-value two-sided:', o_d_t)
# print('Subsample vs Optimal: p-value two-sided:', o_s_t)
# print('Default vs Subsample: p-value two-sided:', s_d_t)

# print('SS mean', SS.mean())
# print('Default mean', Default.mean())
# print('Optimized mean', Optimized.mean())

# # print()
# # print('Default vs Optimal: p-value Optimal greater than Default:', o_d_g)
# # print('Default vs Subsample: p-value Subsample greater than Default:', s_d_g)







