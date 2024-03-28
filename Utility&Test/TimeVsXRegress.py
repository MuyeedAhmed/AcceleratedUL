import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.ticker as ticker

from matplotlib.ticker import FuncFormatter

fontS = 50


def plotAll():
    df_SS = pd.read_csv("Stats/Merged_SS.csv")
    df_SS = df_SS[df_SS["Filename"]!="20_newsgroups.drift_OpenML"]
    
    df_SS["RC"] = df_SS["Row"]*df_SS["Columm"]
    rows = df_SS["Row"].to_numpy()
    column = df_SS["Columm"].to_numpy()
    rc = df_SS["RC"].to_numpy() 
    
    x = np.array(rc).reshape(-1,1)
    
    algos = ["AP", "DBSCAN", "HAC", "SC"]
    colors = ["blue", "salmon", "green", "yellow"]
    
    fig, ax = plt.subplots(figsize=(8, 6))

    for algo, color in zip(algos, colors):
        time = df_SS["Time_"+algo].to_numpy()
        y=time
        
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        residuals = y - y_pred
        r2 = r2_score(y, y_pred)
        print("Original R^2 score:", r2)
        outlier_threshold = np.percentile(np.abs(residuals), 90)
        
        outliers_mask = np.abs(residuals) > outlier_threshold
        
        X_cleaned = x[~outliers_mask]
        y_cleaned = y[~outliers_mask]
        model.fit(X_cleaned, y_cleaned)
        y_pred_cleaned = model.predict(X_cleaned)
        r2_cleaned = r2_score(y_cleaned, y_pred_cleaned)
        print("R^2 score after removing outliers:", r2_cleaned)
        # plt.scatter(x, y, color='blue', label='Original Data')
        plt.plot(x, y_pred, color=color, linewidth=8, label='Regression Line')
        
        plt.scatter(X_cleaned, y_cleaned, marker='o', color=color, label='Cleaned Data')
        plt.xlabel("Points (n) x Features (d)", fontsize=fontS)
        plt.ylabel("Time (s)", fontsize=fontS)
        
        plt.xticks([0, 50000000, 100000000], fontsize=fontS)
        plt.yticks(fontsize=fontS)
        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        plt.grid(False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.legend()
        # plt.title(algo, fontsize=50)
        # plt.savefig('Figures/RuntimeRegression/'+algo+'_LT.pdf', bbox_inches='tight')

    plt.show()
# plotAll()      
        
def plotTimeVsRows(algo):
    print(algo)
    df_SS = pd.read_csv("Stats/Merged_SS.csv")
    df_SS = df_SS[df_SS["Filename"]!="20_newsgroups.drift_OpenML"]
    # df_SS = pd.read_csv("../Stats/Merged_SS.csv")
    
    # df_SS["RC"] = df_SS["Row"]*np.sqrt(df_SS["Columm"])
    df_SS["RC"] = df_SS["Row"]*df_SS["Columm"]
    rows = df_SS["Row"].to_numpy()
    column = df_SS["Columm"].to_numpy()
    rc = df_SS["RC"].to_numpy() 
    time = df_SS["Time_"+algo].to_numpy()


    # plotRegress(rc, time, algo, "Points (n) x sqrt(Features (d))" , "Time (s)")
    # plotRegress(rows, column)
    plotRegress(rows, time, algo, "Points (n) x Features (d)" , "Time (s)")
    # GetAutoEncoder(rows, column)
    # GetXYLinearRegress(rows, column, time)
    
def GetXYLinearRegress(x,y,t):
    X = np.column_stack((x, y))

    model = LinearRegression()
    model.fit(X, t)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    print(mean_squared_error(y, y_pred))
    print(r2)
    
    
def format_ticks(x, pos):
    if x >= 1e6:
        return f'{int(x/1e6)}e6'
    else:
        return f'{int(x)}'



def plotRegress(x, y, algo="", xlabel="", ylabel=""):
    slope, intercept = np.polyfit(x, y, 1)
    print("slope, intercept", slope, intercept)
    regression_line = slope * x + intercept
    r2 = r2_score(y, regression_line)

    '''Poly2'''
    # coefficients = np.polyfit(x, y,2)
    # a, b, c = coefficients
    # x_values = np.linspace(min(x), max(x), 164)
    # regression_line = a * x_values ** 2 + b * x_values + c
    # predicted_y = np.polyval(coefficients, x)
    # r2 = r2_score(y, predicted_y)
    
    print("R2", r2)
    

    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # plt.plot(x, y, 'o')
    # plt.xlabel(xlabel, fontsize=14)
    # plt.ylabel(ylabel, fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.title(algo)

    # plt.plot(x, regression_line, color='red', label='Linear Regression Line')
    # # plt.plot(x_values, regression_line, color='red', label='Quadratic Regression')
    
    # plt.show()
    
    
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.array(x).reshape(-1,1)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    print("Original R^2 score:", r2)
    outlier_threshold = np.percentile(np.abs(residuals), 90)
    
    outliers_mask = np.abs(residuals) > outlier_threshold
    
    X_cleaned = x[~outliers_mask]
    y_cleaned = y[~outliers_mask]
    model.fit(X_cleaned, y_cleaned)
    y_pred_cleaned = model.predict(X_cleaned)
    r2_cleaned = r2_score(y_cleaned, y_pred_cleaned)
    print("R^2 score after removing outliers:", r2_cleaned)
    # plt.scatter(x, y, color='blue', label='Original Data')
    plt.plot(x, y_pred, color='salmon', linewidth=8, label='Regression Line')
    
    plt.scatter(X_cleaned, y_cleaned, marker='o', color='darkblue', label='Cleaned Data')
    # plt.xlabel(xlabel, fontsize=fontS)
    if algo == "AP":
        plt.ylabel(ylabel, fontsize=fontS)
        plt.yticks([0, 500, 1000, 1500, 2000], fontsize=fontS)
    else:
        plt.yticks(fontsize=fontS)
        
    plt.xticks([0, 50000000, 100000000], fontsize=fontS)
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
    # plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.legend()
    plt.title(algo, fontsize=50)
    plt.savefig('Figures/RuntimeRegression/'+algo+'_LT.pdf', bbox_inches='tight')

    plt.show()

    
plotTimeVsRows("AP")
plotTimeVsRows("DBSCAN")
plotTimeVsRows("HAC")
plotTimeVsRows("SC")