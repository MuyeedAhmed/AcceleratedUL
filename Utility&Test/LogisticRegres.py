from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


prev_data = pd.read_csv("Stats/OCSVM_nIter.csv")

X_train = prev_data["rows"].to_numpy()
y_train = prev_data["n_iter"].to_numpy()

X_train = X_train.reshape(-1,1)
reg = LinearRegression().fit(X_train, y_train)

print("Regression Score: ", reg.score(X_train, y_train))

test_data = pd.read_csv("Stats/OCSVM_Incomplete.csv")

X_test = test_data["rows"].to_numpy()

X_test = X_test.reshape(-1,1)

y_test = reg.predict(X_test)

print(y_test)

test_data["Predicted_Iter"] = y_test

test_data["% Done"] = test_data["n_iter"]/test_data["Predicted_Iter"] * 100

print(test_data)


test_data.to_csv("Stats/OCSVM_Incomplete.csv", index=False)