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



def TimeCalc(algo, mode, system):
    folderpath = '../Openml/'
    
    done_files = []
    if os.path.exists("Stats/Time/" + algo + "/"+ system + ".csv") == 0:
        if os.path.isdir("Stats/Time/" + algo + "/") == 0:    
            os.mkdir("Stats/Time/" + algo + "/")
        f=open("Stats/Time/" + algo + "/"+ system + ".csv", "w")
        # f.write('Filename,Row,Columm,Estimated_Time,100,200,300,400,500,600,700,800,900,1000,2000,3000,6000,9000,12000,15000,20000\n')
        f.write('Filename,Row,Columm,Estimated_Time,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,\n')
        # f.write('Filename,Row,Columm,Time,ARI\n')

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
        if filename not in FileList or filename == "2dplanes_OpenML":
            continue
        if filename in done_files:
            print("Already done", filename)
            continue
        runfile(file, filename, algo, mode, system)
        
    times = pd.read_csv("Stats/Time/" + algo + "/"+ system + ".csv")
    
    
    
def runfile(file, filename, algo, mode, system):
    print(filename)
    df = pd.read_csv(file)
    gt = df["class"].to_numpy()
    df = df.drop(columns=["class"])
    row = df.shape[0]
    col = df.shape[1]
    
    
    if row > 110000:
        return
    # if row < 30000:
    #     print("Row:", row)
    #     return
    
    rows = [1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]
    # rows = [100,200,300,400,500,600,700,800,900,1000,2000,3000,6000,9000,12000,15000,20000]
    # rows = [300,600,900,1200,1500,1800]
    # rows = [row]
    times = []
    # ari = -2
    for r in rows:
        print(r, end=' - ')
        X = df[:r]
        n_c = len(np.unique(gt[:r]))
        
        
        t0 = time.time()
        if algo == "AP":
            clustering = AffinityPropagation().fit(X)
        elif algo == "SC":
            clustering = SpectralClustering(n_clusters=n_c,eigen_tol=0.001).fit(X)
            # labels = clustering.labels_
            # ari = adjusted_rand_score(gt, labels)
        elif algo == "DBSCAN":
            clustering = DBSCAN().fit(X)
        else:
            print("Wrong Algo")
            return
        time_ = time.time()-t0
        print(time_)
        times.append(time_)
        # if (r < 1000 and time_ > 10) or time_ > 200:
        #     time_str = ",".join(str(x) for x in times)
            
        #     estimated_time = predict_row_time(rows[:len(times)], times, row)
            
        #     f=open("Stats/Time/" + algo + "/"+ system + ".csv", "a")
        #     f.write(filename+','+str(row)+','+str(col)+','+str(estimated_time)+','+time_str+'\n')
        #     f.close()
        #     return
        
        
    estimated_time = predict_row_time(rows, times, row)
    
    time_str = ",".join(str(x) for x in times)
    f=open("Stats/Time/" + algo + "/"+ system + ".csv", "a")
    f.write(filename+','+str(row)+','+str(col)+','+str(estimated_time)+','+time_str+'\n')
    # f.write(filename+','+str(row)+','+str(col)+','+time_str+','+str(ari)+'\n')
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
    
    fl = ["nomao_OpenML", "OnlineNewsPopularity_OpenML", "2dplanes_OpenML", "mv_OpenML", "numerai28.6_OpenML", "Diabetes130US_OpenML", "BNG(vote)_OpenML", "BNG(2dplanes)_OpenML"]
    for index, row in times.iterrows():
        
        data = row[4:].dropna()
        # if row["Filename"] not in fl:
        #     continue
        x = data.index.values.reshape(-1, 1) 
        x = np.array([list(map(float, item)) for item in x])

        y = data.values.reshape(-1, 1) 
        # y_s = y
        d = 1

        poly_features = PolynomialFeatures(degree=d)
        x_poly = poly_features.fit_transform(x)
    
        model = LinearRegression(fit_intercept=False)
        model.fit(x_poly, y)
        
        y_pred = model.predict(x_poly)
        
        if algo == "AP":
            overestimation_indices = np.where(y_pred > y)[0]
            overestimated_actual = y[overestimation_indices]
            overestimated_predicted = y_pred[overestimation_indices]
    
    
            mse = round(mean_squared_error(overestimated_actual, overestimated_predicted), 1)
            
            if mse > 50:        
                y_s = y*0.92
                model = LinearRegression(fit_intercept=False)
                model.fit(x_poly, y_s)
                
                y_pred = model.predict(x_poly)
        
        # y_corrected = np.copy(y)
        
        # if (y_pred[-2][0]-y[-2][0])/y[-2][0] > 0.1:
        #     y_corrected[-1] *= 1-(y_pred[-2][0]-y[-2][0])/y[-2][0]
        #     # y_corrected[-1] *= 1+(y_pred[-1][0]-y[-1][0])/y[-1][0]

        #     model.fit(x_poly, y_corrected)
    
        #     y_pred = model.predict(x_poly)
            
        
        
        plt.scatter(x, y, label="Data")
        
        plt.plot(x, y_pred, color='red', label=f"Regression Line")
        plt.title(row["Filename"])
        
        plt.legend()
        
        plt.show()
        
        x_test_poly = poly_features.transform(np.array([[row["Row"]]]))
        prediction = model.predict(x_test_poly)[0][0]
        print(round(prediction, 0), "\t", row["Filename"], row["Row"], ":")
    
        times.at[index, "Estimated_Time"] = prediction
        
    # times.to_csv("Stats/Time/" + algo + "/"+ system + "_Est.csv", index=False)
    
    
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def DecisionTree(algo, mode, system):
    times = pd.read_csv("Stats/Time/" + algo + "/"+ system + "_def.csv")    
    gt = pd.read_csv("Stats/" + algo + "/"+ system + ".csv")
    gt = gt[["Filename", "Time"]]
    df = pd.merge(times, gt, on='Filename', how='outer')
    
    # print(df.head(1))
    
    ddf = df.dropna(subset=["Time"])
    
    X = ddf.drop(columns=["Time", "Estimated_Time", "Filename", "2000", "Row"])
    y = ddf["2000"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    tree_regressor = DecisionTreeRegressor(random_state=42)
    tree_regressor.fit(X_train, y_train)
    
    y_pred = tree_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print(y_test, y_pred)
    
def NN(algo, mode, system):
    times = pd.read_csv("Stats/Time/" + algo + "/"+ system + ".csv")
    data = times.drop(columns=["Estimated_Time", "Row", "Filename", "9000", "12000", "15000", "20000"])
    print(data.head())
    

algo = sys.argv[1]
mode = sys.argv[2]
system = sys.argv[3]

TimeCalc(algo, mode, system)

# TimeCalc("SC", "Default", "Louise_test")

# linRegresCalculate("SC", "Default", "Jimmy_def")
# DecisionTree("SC", "Default", "Jimmy")


# NN("AP", "Default", "Jimmy_")


