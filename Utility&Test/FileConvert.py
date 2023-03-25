#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:17:07 2023

@author: muyeedahmed
"""

from scipy.io import arff
import pandas as pd
import numpy as np
import glob

df_csv_append = pd.DataFrame()
arff_files = glob.glob('Dataset/Data_Large/WDBC/*.{}'.format('arff'))

for file in arff_files:

    data = arff.loadarff(file)
    df = pd.DataFrame(data[0])
    
    df["outlier"] = df["outlier"].str.decode("utf-8")

    df["outlier"] = pd.Series(np.where(df.outlier.values == "yes", 1, 0),df.index)

    df = df.drop("id", axis=1)

    # print(df)

    print(sum(df.outlier.values))

    df_csv_append = pd.concat([df_csv_append, df])

# df_csv_append = df_csv_append.drop_duplicates()
df_csv_append.to_csv('Dataset/WDBC.csv', index=False)


print(df_csv_append.groupby(df_csv_append.columns.tolist(),as_index=False).size())