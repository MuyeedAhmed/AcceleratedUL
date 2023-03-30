#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 05:36:02 2023

@author: muyeedahmed
"""

import pandas as pd
file_path = "../Dataset/kddcup.data"
# above .data file is comma delimited
file_path = pd.read_csv(file_path, delimiter=",", header=None)

file_path.to_csv("../Dataset/kddcup.csv",index=False)