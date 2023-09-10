import os
import pandas as pd
import numpy as np
import time
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
import sys
import glob

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score



def TimeCalc(algo, mode, system):
    folderpath = '../Openml/'
    
    done_files = []
    if os.path.exists("Stats/Time/" + algo + "/"+ system + ".csv") == 0:
        if os.path.isdir("Stats/Time/" + algo + "/") == 0:    
            os.mkdir("Stats/Time/" + algo + "/")
        f=open("Stats/Time/" + algo + "/"+ system + ".csv", "w")
        # f.write('Filename,Row,Columm,Estimated_Time,100,200,300,400,500,600,700,800,900,1000,2000,3000,6000,9000,12000,15000,20000\n')
        f.write('Filename,Row,Columm,Estimated_Time,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000\n')
        f.close()
    else:
        done_files = pd.read_csv("Stats/Time/" + algo + "/"+ system + ".csv")
        done_files = done_files["Filename"].to_numpy()

    
    master_files = glob.glob(folderpath+"*.csv")
    
    FileList = pd.read_csv("MemoryStats/FileList.csv")
    FileList = FileList["Filename"].to_numpy()
    
    for file in master_files:
        filename = file.split("/")[-1]
        filename = filename[:-4]
        if filename not in FileList:
            continue
        if filename in done_files:
            print("Already done", filename)
            continue
        runfile(file, filename, algo, mode, system)
        
    times = pd.read_csv("Stats/Time/" + algo + "/"+ system + ".csv")
    
    
    
def runfile(file, filename, algo, mode, system):
    print(filename)
    df = pd.read_csv(file)
    df = df.drop(columns=["class"])
    row = df.shape[0]
    col = df.shape[1]
    
    if row > 110000:
        return
    # if row < 30000:
    #     print("Row:", row)
    #     return
    
    rows = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
    # rows = [100,200,300,400,500,600,700,800,900,1000,2000,3000,6000,9000,12000,15000,20000]
    # rows = [300,600,900,1200,1500,1800]
    times = []
    for r in rows:
        print(r, end=' - ')
        X = df[:r]
        
        t0 = time.time()
        if algo == "AP":
            clustering = AffinityPropagation().fit(X)
        elif algo == "SC":
            clustering = SpectralClustering(eigen_solver="lobpcg").fit(X)
        elif algo == "DBSCAN":
            clustering = DBSCAN().fit(X)
        else:
            print("Wrong Algo")
            return
        time_ = time.time()-t0
        print(time_)
        times.append(time_)
        if (r < 1000 and time_ > 10) or time_ > 200:
            time_str = ",".join(str(x) for x in times)
            
            estimated_time = predict_row_time(rows[:len(times)], times, row)
            
            f=open("Stats/Time/" + algo + "/"+ system + ".csv", "a")
            f.write(filename+','+str(row)+','+str(col)+','+str(estimated_time)+','+time_str+'\n')
            f.close()
            return
        
        
    estimated_time = predict_row_time(rows, times, row)
    
    time_str = ",".join(str(x) for x in times)
    f=open("Stats/Time/" + algo + "/"+ system + ".csv", "a")
    f.write(filename+','+str(row)+','+str(col)+','+str(estimated_time)+','+time_str+'\n')
    f.close()
        
        
def predict_row_time(X, Y, row):
    X = np.array(X)
    Y = np.array(Y)

    x = X.reshape(-1, 1)
    y = Y.reshape(-1, 1)
    
    d = 2

    poly_features = PolynomialFeatures(degree=d)
    x_poly = poly_features.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_poly, y)
    
    x_test_poly = poly_features.transform(np.array([[row]]))
    prediction = model.predict(x_test_poly)[0][0]
    
    return prediction
        
def linRegresCalculate(algo, mode, system):
    times = pd.read_csv("Stats/Time/" + algo + "/"+ system + ".csv")    
    
    for index, row in times.iterrows():
        data = row[4:].dropna()
        
        x = data.index.values.reshape(-1, 1) 
        x = np.array([list(map(float, item)) for item in x])

        y = data.values.reshape(-1, 1) 
        
        d = 2

        poly_features = PolynomialFeatures(degree=d)
        x_poly = poly_features.fit_transform(x)
        
        model = LinearRegression()
        model.fit(x_poly, y)
        
        y_pred = model.predict(x_poly)

        # mse = mean_squared_error(y, y_pred)
        # r2 = r2_score(y, y_pred)

        # print(f"MSE: {mse}")
        # print(f"R2: {r2}")
        
        
        plt.scatter(x, y, label="Data")
        
        plt.plot(x, y_pred, color='red', label=f"Regression Line")
        plt.title(row["Filename"])
        
        plt.legend()
        
        plt.show()
        
        x_test_poly = poly_features.transform(np.array([[row["Row"]]]))
        prediction = model.predict(x_test_poly)[0][0]
        print(row["Filename"], row["Row"], ":", prediction)
    
        
        # y = y.flatten()
        # print(y)
        # plt.scatter(x, y, label="Data Points")
        # best_degree_poly_features = PolynomialFeatures(degree=d)
        # time_best_degree_poly = best_degree_poly_features.fit_transform(x)
        # best_degree_model = LinearRegression()
        # best_degree_model.fit(time_best_degree_poly, y)
        # cost_pred_best_degree = best_degree_model.predict(time_best_degree_poly)
        # plt.plot(x, cost_pred_best_degree, color='red', label="Regression Line")
        # plt.xlabel("Time")
        # plt.ylabel("Cost")
        # plt.legend()
        # plt.title("Polynomial Regression")
        # plt.show()    
        
def NN(algo, mode, system):
    times = pd.read_csv("Stats/Time/" + algo + "/"+ system + ".csv")
    data = times.drop(columns=["Estimated_Time", "Row", "Filename", "9000", "12000", "15000", "20000"])
    print(data.head())
    

# algo = sys.argv[1]
# mode = sys.argv[2]
# system = sys.argv[3]

# TimeCalc(algo, mode, system)

TimeCalc("SC", "Default", "Louise")

# linRegresCalculate("AP", "Default", "Jimmy_")

# NN("AP", "Default", "Jimmy_")


