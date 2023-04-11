#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 03:04:33 2023

@author: muyeedahmed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
n_samples = 4000000
centers = [(5, 5), (5, -5), (-5, 5), (-5, -5)]
X = np.zeros((n_samples, 2))
for i, center in enumerate(centers):
    X[i * (n_samples // len(centers)): (i + 1) * (n_samples // len(centers)), :] = np.random.randn(n_samples // len(centers), 2) + np.array(center)

# Create DataFrame
df = pd.DataFrame(X, columns=['x', 'y'])

# Add cluster column
df['target'] = np.repeat(range(len(centers)), n_samples // len(centers))

# df.to_csv("Temp/Fake.csv", index=False)
# # Plot DataFrame
colors = ['r', 'g', 'b', 'y']
fig, ax = plt.subplots()
for i, cluster in enumerate(range(len(centers))):
    df_cluster = df[df['target'] == cluster]
    ax.scatter(df_cluster['x'], df_cluster['y'], c=colors[i], label=f'Cluster {cluster}')
ax.legend()
plt.show()
