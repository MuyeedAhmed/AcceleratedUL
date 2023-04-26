#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:26:27 2023

@author: muyeedahmed
"""

import pandas as pd
from scipy.stats import mannwhitneyu


df = pd.read_csv("Stats/GMM.csv")

Default = df["ARI_WD"].to_numpy()

SS = df["ARI_SS"]

Optimized = df["ARI_WO"]


_, o_d_t = mannwhitneyu(x=Optimized, y=Default, alternative = 'two-sided', use_continuity=True)
_, s_d_t = mannwhitneyu(x=SS, y=Default, alternative = 'two-sided', use_continuity=True)


_, o_d_g = mannwhitneyu(x=Optimized, y=Default, alternative = 'greater', use_continuity=True)
_, s_d_g = mannwhitneyu(x=SS, y=Default, alternative = 'greater', use_continuity=True)


print('Default vs Optimal: p-value two-sided:', o_d_t)
print('Default vs Subsample: p-value two-sided:', s_d_t)
print()
print('Default vs Optimal: p-value Optimal greater than Default:', o_d_g)
print('Default vs Subsample: p-value Subsample greater than Default:', s_d_g)


