import pandas as pd


import openml
import openml.config

openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


def saveOpenMLFile():
    dataset_list = openml.datasets.list_datasets()
    
    for key, ddf in dataset_list.items():
        if "NumberOfInstances" in ddf:
            if ddf["NumberOfInstances"] >= 10000:
                
                filename = ddf["name"]+"_OpenML" 
                filename = filename.replace(",", "_COMMA_")
            
                print(ddf["name"])
                did =  ddf["did"]
   
                dataset = openml.datasets.get_dataset(did)
                 
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    dataset_format="array", target=dataset.default_target_attribute
                    )
                df = pd.DataFrame(X)
                
                r = df.shape[0]
                c = df.shape[1]
                
                df["class"] = y
                is_numeric = df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
                
                if all(is_numeric) and c >= 10:                
                    df.to_csv("../Openml/"+filename+".csv", index=False)
                    
saveOpenMLFile()      
    