#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:26:27 2023

@author: muyeedahmed
"""

import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt


df1 = pd.read_csv("Stats/AP.csv")
df2 = pd.read_csv("Stats/AP_Best_Subsample_Run_ModeB.csv")



df = df1.merge(df2, on='Filename', how='inner')

Default = df["ARI_WD"]

SS = df["ARI_SS"]

Optimized = df["ARI_WO"]

df["Difference"] = Default - SS

df = df.sort_values('Difference')

diff = df["Difference"].to_numpy()

plt.figure(0)

plt.plot(diff)

plt.figure(1)
SS.plot.kde()
Default.plot.kde()

plt.figure(2)
SS.reset_index().plot.scatter(x='index', y='ARI_SS', color='red')
Default.reset_index().plot.scatter(x='index', y='ARI_WD', color='blue')



# stat, p = ttest_ind(group1, group2)

_, o_d_t = ttest_ind(Optimized, Default)
_, o_s_t = ttest_ind(Optimized, SS)
_, s_d_t = ttest_ind(SS, Default)


# _, o_d_t = mannwhitneyu(x=Optimized, y=Default, alternative = 'two-sided')
# _, s_d_t = mannwhitneyu(x=SS, y=Default, alternative = 'two-sided')


# _, o_d_g = mannwhitneyu(x=Optimized, y=Default, alternative = 'greater', use_continuity=True)
# _, s_d_g = mannwhitneyu(x=SS, y=Default, alternative = 'greater', use_continuity=True)


print('Default vs Optimal: p-value two-sided:', o_d_t)
print('Subsample vs Optimal: p-value two-sided:', o_s_t)
print('Default vs Subsample: p-value two-sided:', s_d_t)

print('SS mean', SS.mean())
print('Default mean', Default.mean())
print('Optimized mean', Optimized.mean())

# print()
# print('Default vs Optimal: p-value Optimal greater than Default:', o_d_g)
# print('Default vs Subsample: p-value Subsample greater than Default:', s_d_g)


