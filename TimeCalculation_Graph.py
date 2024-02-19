import os
import pandas as pd
import numpy as np
import time
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
import sys
import glob
from sklearn.metrics.cluster import adjusted_rand_score


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.ticker as ticker






def draw_AP_time(algo, mode, system):
    times_def = pd.read_csv("Stats/Time/" + algo + "/"+ system + ".csv")    
    times_ss = pd.read_csv("Stats/Time/" + algo + "/M2_SS.csv")    
    # times_001 = pd.read_csv("Stats/Time/" + algo + "/"+ system + "_0.001.csv")    
    # times_00001 = pd.read_csv("Stats/Time/" + algo + "/"+ system + "_0.00001.csv")    
                
    
    # fl = ["nomao_OpenML", "OnlineNewsPopularity_OpenML", "2dplanes_OpenML", "mv_OpenML", "numerai28.6_OpenML", "Diabetes130US_OpenML", "BNG(vote)_OpenML", "BNG(2dplanes)_OpenML"]
    
    for index, row in times_def.iterrows():
        filename = row["Filename"]
        if filename != "numerai28.6_OpenML":
            continue
        
        data_def = row[4:30].dropna()
        x = data_def.index.values.reshape(-1, 1) 
        x_def = np.array([list(map(float, item)) for item in x])
        y_def = data_def.values.reshape(-1, 1) 
        
       
        
        data_ss = times_ss.loc[times_ss['Filename'] == filename]
        data_ss = data_ss.to_numpy()[0]
        y_ss = data_ss[4:12]
        # x_ss = [1000,2000,3000,6000,9000,12000,15000,20000,50000,65000]
        x_ss = [1000,2000,3000,6000,9000,12000,15000,20000]
        

        plt.plot(x_ss, y_ss, label="ACE", color='darkorange')
        plt.plot(x_def, y_def, label="Default", color='steelblue')

        
        # plt.plot(x, y_pred, color='red', label=f"Regression Line")
        plt.grid(False)
        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

        plt.xlabel("# Points")
        plt.ylabel("Time (seconds)")
        plt.title("Affinity Propagation")
        # plt.title(row["Filename"])
        plt.legend()
        plt.savefig('Figures/Time_AP_Intro.pdf', bbox_inches='tight')
        plt.show()

def draw_sc_memory():
    rows = [i for i in range(10000,1000001,10000)]       
    rows_observed = [i for i in range(10000,150001,10000)]
    rows_predicted = [i for i in range(150000,1000001,10000)]
    # memory_d = (np.square(rows)*4*4)/1000000000
    
    memory_d = [((x**2)*4*8)/1000000000  for x in rows]
    memory_s = [(((x/100)**2)*4*8*10)/1000000000  for x in rows]
    memory_obs = [((x**2)*4*8)/1000000000  for x in rows_observed]
    memory_pred = [((x**2)*4*8)/1000000000  for x in rows_predicted]
    # print(memory_d[0])
    # print(memory_d[-1])
    # print(memory_s[0])
    # print(memory_s[-1])
    
    # plt.plot(rows, memory_d, label="Default")
    plt.plot(rows, memory_s, label="ACE", color='darkorange')
    plt.axvline(x = 150001, color = 'black', linestyle = ':')
    plt.text(190001, 7500, 'Maximum observed memory requirement due to resource limitation', 
         color = 'black', rotation = 90, 
         rotation_mode = 'anchor')
    plt.plot(rows_observed, memory_obs, label="Default (Observed)", color='steelblue')
    plt.plot(rows_predicted, memory_pred, label="Default (Predicted)", color='#949494', linestyle='--')
    plt.grid(False)
    plt.xlabel("# Points")
    plt.ylabel("Memory (GB)")
    plt.title("Spectral Clustering")
    # plt.title(row["Filename"])
    plt.legend()
    
    import matplotlib.ticker as ticker    
    def format_with_commas(x, pos):
        return "{:,}".format(int(x))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_with_commas))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_with_commas))

    # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

    plt.savefig('Figures/Memory_SC_Intro.pdf', bbox_inches='tight')
    plt.show()
    

def draw_SC(algo, mode, system):
    times_def = pd.read_csv("Stats/Time/" + algo + "/"+ system + "_def.csv")    
    times_01 = pd.read_csv("Stats/Time/" + algo + "/"+ system + "_0.01.csv")    
    times_001 = pd.read_csv("Stats/Time/" + algo + "/"+ system + "_0.001.csv")    
    times_00001 = pd.read_csv("Stats/Time/" + algo + "/"+ system + "_0.00001.csv")    
                
    
    # fl = ["nomao_OpenML", "OnlineNewsPopularity_OpenML", "2dplanes_OpenML", "mv_OpenML", "numerai28.6_OpenML", "Diabetes130US_OpenML", "BNG(vote)_OpenML", "BNG(2dplanes)_OpenML"]
    
    for index, row in times_def.iterrows():
        filename = row["Filename"]
        
        
        data_def = row[4:].dropna()
        x = data_def.index.values.reshape(-1, 1) 
        x_def = np.array([list(map(float, item)) for item in x])
        y_def = data_def.values.reshape(-1, 1) 
        
        data_01 = times_01.loc[times_01['Filename'] == filename]
        if data_01.shape[0] == 0:
            print(filename , "01 unavailable")
            continue
        data_01 = data_01.to_numpy()[0]
        y_01 = data_01[5:]
        x_01 = [i for i in range(1100,3100,100)]

        data_001 = times_001.loc[times_001['Filename'] == filename]
        if data_001.shape[0] == 0:
            print(filename , "001 unavailable")
            continue
        data_001 = data_001.to_numpy()[0]
        y_001 = data_001[5:]
        x_001 = [i for i in range(1100,3100,100)]
        
        data_00001 = times_00001.loc[times_00001['Filename'] == filename]
        if data_00001.shape[0] == 0:
            print(filename , "00001 unavailable")
            continue
        data_00001 = data_00001.to_numpy()[0]
        y_00001 = data_00001[5:]
        x_00001 = [i for i in range(1100,3100,100)]
        
        
        # return
        # data_01 = row[5:].dropna()
        
        
        
        # # if row["Filename"] not in fl:
        # #     continue
    
        # x = data.index.values.reshape(-1, 1) 
        # x = np.array([list(map(float, item)) for item in x])

        # y = data.values.reshape(-1, 1) 
        # # y_s = y
        # d = 1

        # poly_features = PolynomialFeatures(degree=d)
        # x_poly = poly_features.fit_transform(x)
    
        # model = LinearRegression(fit_intercept=False)
        # model.fit(x_poly, y)
        
        # y_pred = model.predict(x_poly)
        
        
        plt.scatter(x_def, y_def, label="Default")
        plt.scatter(x_01, y_01, label="0.01")
        plt.scatter(x_001, y_001, label="0.001")
        plt.scatter(x_00001, y_00001, label="0.00001")

        
        # plt.plot(x, y_pred, color='red', label=f"Regression Line")
        
        plt.title(row["Filename"])
        plt.legend()
        plt.show()
        
        # x_test_poly = poly_features.transform(np.array([[row["Row"]]]))
        # prediction = model.predict(x_test_poly)[0][0]
        # print(round(prediction, 0), "\t", row["Filename"], row["Row"], ":")
    
        # times.at[index, "Estimated_Time"] = prediction

draw_sc_memory()
# draw_AP_time("AP", "Default", "Jimmy_EST")