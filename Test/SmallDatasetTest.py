#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 07:18:33 2023

@author: muyeedahmed
"""

from sklearn.cluster import DBSCAN
import numpy as np
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80], [1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80], [1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(algorithm="brute").fit(X)
clustering.labels_