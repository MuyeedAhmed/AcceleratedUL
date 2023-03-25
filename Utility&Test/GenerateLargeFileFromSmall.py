#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:56:43 2023

@author: muyeedahmed
"""

from scipy.io import arff
import pandas as pd
import numpy as np
import glob
import os

df_csv_append = pd.DataFrame()
small_files = glob.glob('Dataset/Small/*.{}'.format('csv'))

for file in small_files:
    size = os.path.getsize(file)
    
    df_csv_append = pd.DataFrame()
    df = pd.read_csv(file)
    iteration = int(10000000/size)
    for _ in range(iteration):
        df_csv_append = pd.concat([df_csv_append, df])
    df_csv_append.to_csv(file, index=False)


