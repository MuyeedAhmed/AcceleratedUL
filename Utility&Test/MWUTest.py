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


df = pd.read_csv("Stats/AP.csv")

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
_, s_d_t = ttest_ind(SS, Default)


# _, o_d_t = mannwhitneyu(x=Optimized, y=Default, alternative = 'two-sided')
# _, s_d_t = mannwhitneyu(x=SS, y=Default, alternative = 'two-sided')


# _, o_d_g = mannwhitneyu(x=Optimized, y=Default, alternative = 'greater', use_continuity=True)
# _, s_d_g = mannwhitneyu(x=SS, y=Default, alternative = 'greater', use_continuity=True)


print('Default vs Optimal: p-value two-sided:', o_d_t)
print('Default vs Subsample: p-value two-sided:', s_d_t)
print()
print('Default vs Optimal: p-value Optimal greater than Default:', o_d_g)
print('Default vs Subsample: p-value Subsample greater than Default:', s_d_g)


