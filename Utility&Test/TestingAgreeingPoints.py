#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 05:57:07 2023

@author: muyeedahmed
"""
from itertools import combinations

def find_agreeing_points(clustering_outputs):
    n = len(clustering_outputs)
    agreeing_points = set(range(len(clustering_outputs[0])))

    for i, j in combinations(range(n), 2):
        points_i = set([k for k, c in enumerate(clustering_outputs[i]) if c == 1])
        points_j = set([k for k, c in enumerate(clustering_outputs[j]) if c == 1])
        agreeing_points = agreeing_points.intersection(points_i.intersection(points_j))

    return agreeing_points



clustering_outputs = [[1,1,2,2,3], [2,2,1,1,1], [3,3,1,2,1]]

print(find_agreeing_points(clustering_outputs))