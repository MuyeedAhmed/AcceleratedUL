import pandas as pd
import os
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
import numpy as np
"""GMM"""

# df1 = pd.read_csv("Stats/GMM.csv")
# ablation_old = pd.read_csv("Stats/GMM_Ablation.csv")
# ablation_small = pd.read_csv("Stats/GMM_Ablation_Small.csv")
# df2 = pd.concat([ablation_old, ablation_small], axis=0).reset_index(drop=True)
# df2 = df2.dropna(axis='columns')
# df = df1.merge(df2, on='Filename', how='inner')

"""AP"""

# df1 = pd.read_csv("Stats/AP.csv")
# ablation_old = pd.read_csv("Stats/AP_Ablation.csv")
# ablation_small = pd.read_csv("Stats/AP_Ablation_Small.csv")
# df2 = pd.concat([ablation_old, ablation_small], axis=0).reset_index(drop=True)
# df2 = df2.dropna(axis='columns')
# df = df1.merge(df2, on='Filename', how='inner')

"""HAC"""
# df1 = pd.read_csv("Stats/HAC_Default.csv")
# ablation = pd.read_csv("Stats/HAC_Ablation.csv")
# df = df1.merge(ablation, on='Filename', how='inner')

"""DBSCAN"""
# df1 = pd.read_csv("Stats/DBSCAN_Default.csv")
# # ablation = pd.read_csv("Stats/DBSCAN_Ablation_Small.csv")
# ablation = pd.read_csv("Stats/DBSCAN_Ablation_NoAnomaly.csv")
# df = df1.merge(ablation, on='Filename', how='inner')


class EvaluateOutcome:
    def __init__(self, algo):
        self.algorithm = algo
        self.df = pd.DataFrame()
        
        self.bestCombination = ""
        
        self.ari_all_data = []
        
        self.DeterParamComp = ["AP", "KM", "DBS", "HAC", "INERTIA", "AVG"]
        self.RerunModes = ["A", "B"]
        self.MergeModes = ["Distance", "DistanceRatio", "ADLOF", "ADIF", "ADEE", "ADOCSVM"]

        # self.DeterParamComp = ["AP" "KM", "DBS", "HAC", "AVG"]
        # self.RerunModes = ["A", "B"]
        # self.MergeModes = ["Distance", "DistanceRatio", "ADLOF"]

    def readStatFiles(self):
        
        
        # if os.path.exists("Stats/"+self.algorithm+"_Ablation_Small.csv"):
        #     ablation_small = pd.read_csv("Stats/"+self.algorithm+"_Ablation_Small.csv")
        #     self.df = pd.concat([self.df, ablation_small], axis=0).reset_index(drop=True)
        # if os.path.exists("Stats/"+self.algorithm+"_Ablation.csv"):
        #     ablation = pd.read_csv("Stats/"+self.algorithm+"_Ablation.csv")
        #     self.df = pd.concat([self.df, ablation], axis=0).reset_index(drop=True)        
        
        if os.path.exists("Stats/"+self.algorithm+"_Ablation_NoAnomaly.csv"):
            ablation_noA = pd.read_csv("Stats/"+self.algorithm+"_Ablation_NoAnomaly.csv")
            self.df = pd.concat([self.df, ablation_noA], axis=0).reset_index(drop=True)        
        try:
            default = pd.read_csv("Stats/"+self.algorithm+"_Default.csv")
        except:
            default = pd.read_csv("Stats/"+self.algorithm+".csv")
        print(self.df.columns)
        print(default.columns)
        
        self.df = self.df.merge(default, on='Filename', how='inner')
        
        return self.df

    def getBestMode(self):
        stats = pd.DataFrame(columns=['DeterParamComp', 'RerunModes', 'MergeModes', 'Time', 'ARI'])
        maxTime = 0
        maxTimeColumn = ""
        maxARI = -1
        maxARIColumn = ""
        maxARITime = 0
        Time = []
        ARI = []
        self.ari_all_data = []
        for dpc in self.DeterParamComp:
            for rm in self.RerunModes:
                for mm in self.MergeModes:
                    try:
                        columnNameT = "Time_"+dpc+"_"+rm+"_"+mm
                        columnNameA = "ARI_"+dpc+"_"+rm+"_"+mm
                        
                        self.ari_all_data.append(self.df[columnNameA].to_numpy())
                        
                        t = np.mean(self.df[columnNameT].to_numpy())
                        a = np.mean(self.df[columnNameA].to_numpy())
                        Time.append(t)
                        ARI.append(a)
    
                        stats.loc[-1] = [dpc, rm, mm, t, a]
                        stats.index = stats.index + 1  
                        if t > maxTime:
                            maxTime = t
                            maxTimeColumn = columnNameT
                        if a > maxARI:
                            maxARI = a
                            maxARIColumn = columnNameA
                            maxARITime = t
                    except:
                        pass
        self.bestCombination = maxARIColumn
        
        self.df["Diff"] = self.df["ARI_WD"] - self.df[self.bestCombination]
        
        print("Subsampling:")
        print("\tTime", maxARITime)
        print("\tARI: ", maxARI)
        print("\tCombination", maxARIColumn)
        
        return stats, maxARIColumn
 
    def printDefaultMean(self, df):
        ARI_Default = df["ARI_WD"]
        Time_Default = df["Time_WD"]
        
        print(f"Default:\n\tTime:{np.mean(Time_Default.to_numpy())}\n\tARI:{np.mean(ARI_Default.to_numpy())}")

    def printCorrelation(self, df):
        print("Correlation (Default vs Row): ", df["Shape_R_x"].corr(df["ARI_WD"]))
        print("Correlation (Default vs Column): ", df["Shape_C_x"].corr(df["ARI_WD"]))
        if "Diff" in df:
            print("Correlation (Difference vs Row): ", df["Diff"].corr(df["ARI_WD"]))
            print("Correlation (Difference vs Column): ", df["Diff"].corr(df["ARI_WD"]))   
 
    def drawDefaultVSBest(self, df):
        plt.figure(0)

        default = df["ARI_WD"].to_numpy()
        default.sort()
        ss = df[self.bestCombination].to_numpy()
        ss.sort()
        
        plt.plot(default, '.', color='red')
        plt.plot(ss, '.', color='blue')
        plt.title("Sorted")
        plt.xlabel("Datasets")
        plt.ylabel("ARI")
        plt.show()

        """ Sorted - Default """
        sortedDf = df
        sortedDf = sortedDf.sort_values(by=[self.bestCombination])
        sortedDf = sortedDf.sort_values(by=["ARI_WD"])
        plt.figure(1)

        default = sortedDf["ARI_WD"].to_numpy()
        ss = sortedDf[self.bestCombination].to_numpy()
        
        plt.plot(default, '.', color='red')
        plt.plot(ss, '.', color='blue')
        plt.title("Sorted - Default")
        plt.xlabel("Datasets")
        plt.ylabel("ARI")
        plt.show()


    def drawDelta(self, df):
        plt.figure()
        delta = df["Diff"].to_numpy()
        delta.sort()
        
        plt.plot(delta)
        plt.axhline(y = 0, color = 'r', linestyle = '-')

        plt.xlabel("Datasets")
        plt.ylabel("ARI")
        plt.show()
    
        
    def groupByModes_Plot(self, stats, level):
        # stats, _ = self.getBestMode()
        
        groups = stats.groupby(level)
        fig, ax = plt.subplots()
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            ax.plot(group.ARI, group.Time, marker='o', linestyle='', ms=5, label=name)
        ax.legend()
        plt.xlabel("ARI")
        plt.ylabel("Time")
        plt.show()
        



evalModes = EvaluateOutcome("DBSCAN")
evalModes.readStatFiles()
stats, bestMode = evalModes.getBestMode()
evalModes.printDefaultMean(evalModes.df)
evalModes.printCorrelation(evalModes.df)
evalModes.drawDefaultVSBest(evalModes.df)

evalModes.drawDelta(evalModes.df)

evalModes.groupByModes_Plot(stats, "DeterParamComp")
evalModes.groupByModes_Plot(stats, "RerunModes")
evalModes.groupByModes_Plot(stats, "MergeModes")

# df = self.df[self.df["Shape_R_x"] > 7000]





# stats = pd.DataFrame(columns=['DeterParamComp', 'RerunModes', 'MergeModes', 'Time', 'ARI'])

# maxTime = 0
# maxTimeColumn = ""
# maxARI = -1
# maxARIColumn = ""
# maxARITime = 0
# Time = []
# ARI = []

# ari_all_data = []

# for dpc in DeterParamComp:
#     for rm in RerunModes:
#         for mm in MergeModes:
#             columnNameT = "Time_"+dpc+"_"+rm+"_"+mm
#             columnNameA = "ARI_"+dpc+"_"+rm+"_"+mm
            
#             ari_all_data.append(df[columnNameA].to_numpy())
#             a = df[columnNameA].to_numpy()
#             a.sort()
#             plt.plot(a, '.', color='blue')
#             # plt.plot(df[columnNameA].to_numpy(), ".")
#             t = np.mean(df[columnNameT].to_numpy())
#             a = np.mean(df[columnNameA].to_numpy())
#             Time.append(t)
#             ARI.append(a)

#             stats.loc[-1] = [dpc, rm, mm, t, a]
#             stats.index = stats.index + 1  
#             if t > maxTime:
#                 maxTime = t
#                 maxTimeColumn = columnNameT
#             if a > maxARI:
#                 maxARI = a
#                 maxARIColumn = columnNameA
#                 maxARITime = t

# # print(ari_all_data)
# plt.plot(ari_all_data)
# plt.xlabel("ARI")
# # plt.ylabel("Time")
# fig, ax = plt.subplots()
# ax.boxplot(ari_all_data, positions=Time, widths=0.1, vert=False)
# plt.yticks([])


