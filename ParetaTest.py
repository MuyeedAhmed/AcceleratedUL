import matplotlib.pyplot as plt
import numpy as np

def pf(algo):
    data = datareturn(algo)
    
    if algo == "AP":
        referees = ['KM', 'DBS', 'HAC']
    elif algo == "HAC":
        referees = ['KM', 'DBS', 'AP']
    
    for system in referees:
        system_data = data[data['Referee'] == system]
        
        system_data = system_data.sort_values(by=['Normalized_Time', 'ARI'])
        
        pareto_frontier = np.full(system_data.shape[0], True)
        
        for i in range(system_data.shape[0]):
            if i == 0:
                max_accuracy = system_data.iloc[i]['ARI']
            else:
                if system_data.iloc[i]['ARI'] >= max_accuracy:
                    max_accuracy = system_data.iloc[i]['ARI']
                else:
                    pareto_frontier[i] = False
        
        pareto_data = system_data[pareto_frontier]
        
        plt.scatter(system_data['Normalized_Time'], system_data['ARI'], label=f'{system} All')
        plt.plot(pareto_data['Normalized_Time'], pareto_data['ARI'], label='')
        plt.xlabel('Normalized_Time')
        plt.ylabel('ARI')
        plt.legend()
    plt.show()

def scatter(algo):
    data = datareturn(algo)
    
    if algo == "AP":
        referees = ['KM', 'DBS', 'HAC']
    elif algo == "HAC":
        referees = ['KM', 'DBS', 'AP']
        
        
    for referee in referees:
        referee_data = data[data['Referee'] == referee]
        
        avg_time = referee_data['Normalized_Time'].mean()
        avg_ari = referee_data['ARI'].mean()
        if referee == "KM":
            color = "green"
        elif referee == "DBS":
            color = "red"
        elif referee == "HAC":
            color = "blue"
        elif referee == "AP":
            print(referee_data['ARI'])
            color = "purple"
            
        plt.scatter(referee_data['Normalized_Time'], referee_data['ARI'], label=f'{referee}',color=color, marker='.', alpha=0.1)
        plt.scatter(avg_time, avg_ari, color = color, marker="^", label=f'{referee} Average')
        plt.xlabel('Normalized_Time')
        plt.ylabel('ARI')
        plt.legend()
    plt.show()



def datareturn(algo):
    df = pd.read_csv("Stats/Ablation/Ablation_RefereeClAlgo_"+algo+".csv")
    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    df["ARI"] = scaler.fit_transform(df[["ARI"]])
    # Temp    
    df = df[df['ARI'] != 0]
    df["ARI"] = -np.log(df[["ARI"]])

    grouped = df.groupby('Referee')

    df['Normalized_Time'] = df.groupby('Referee')['Time'].transform(lambda x: (x/x.max()))
    df = df.dropna()
    # df['ARI_Normalized_Time'] = df['ARI'] * df['Normalized_Time']
    # print(df)
    return df
    
# scatter("HAC")
pf("HAC")
