import openml
from collections import OrderedDict


openml.config.apikey = '311e9ca589cd8291d0f4f67c7d0ba5de'


dataset_list = openml.datasets.list_datasets(size=1000000000)

# filtered_dict = OrderedDict((key, value) for key, value in dataset_list.items() if value["name"] == "electricity_prices_ICON")

count = 0
for key, ddf in dataset_list.items():
    if "NumberOfInstances" in ddf:
        if ddf["NumberOfInstances"] > 1000000:
            print(ddf)
            count+=1
            id_ =  ddf["did"]
            dataset = openml.datasets.get_dataset(id_)
            
            # print(ddf["NumberOfInstances"])
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
                )
            eeg = pd.DataFrame(X)
            eeg["class"] = y
            is_numeric = eeg.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
            print(all(is_numeric))
            break
# print(filtered_dict)
# print(len(filtered_dict))

print(count)

# datalist = pd.DataFrame.from_dict(dataset_list, orient="index")
# datalist = datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

# datalist[datalist.NumberOfInstances > 10000].sort_values(["NumberOfInstances"]).head(n=20)

# dataset = openml.datasets.get_dataset(1471)
# # df = dataset_list.query("anneal")

# print(dataset)
