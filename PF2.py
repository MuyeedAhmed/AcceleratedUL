import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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

df = datareturn("AP")
df = datareturn("HAC")

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
df['Time_Range'] = pd.cut(df['Normalized_Time'], bins=bins)
min_ari_df = df.groupby(['Time_Range', 'Referee'])['ARI'].min().reset_index()
min_ari_df['midpoint'] = min_ari_df['Time_Range'].apply(lambda x: (x.left + x.right) / 2)

fig, ax = plt.subplots()
algorithms = min_ari_df['Referee'].unique()

colors = plt.cm.get_cmap('rainbow', len(algorithms))

for i, algorithm in enumerate(algorithms):
    # Filter the DataFrame for each algorithm
    df_filtered = min_ari_df[min_ari_df['Referee'] == algorithm]
    ax.plot(df_filtered['midpoint'], df_filtered['ARI'], marker='s', color=colors(i), label=algorithm)

ax.set_xlabel('Normalized Time')
ax.set_ylabel('ARI')
ax.legend()
plt.show()


plt.show()
print(min_ari_df)


    

