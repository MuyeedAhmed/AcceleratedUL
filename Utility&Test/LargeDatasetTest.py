#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 05:45:03 2023

@author: muyeedahmed
"""

import pandas as pd
import numpy as np
import random

df = pd.DataFrame([i*random.randrange(10) for i in np.random.rand(1500000, 50)])
#df = pd.DataFrame(np.random.rand(1000000, 45))
print(df.head())
from sklearn.cluster import DBSCAN
print("start")
c = DBSCAN(algorithm="brute").fit(df)
print(len(c.labels_))

