import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree

df = pd.read_csv("../data/first_round_training_data.csv")
df_test = pd.read_csv("../data/first_round_testing_data.csv")
#parameter *10 attribute *10 Quality_label Fail Pass Good Excellent
quality_map = {"Fail":3,"Pass":2,"Good":1,'Excellent':0}
df["Quality_label"] = df["Quality_label"].map(quality_map)
df_train = df.iloc[:,0:10]
Y_train = df["Quality_label"].to_numpy()
para10 = df_test.pop("Parameter10")
df_test["Parameter10"] = para10
test_group = df_test.pop("Group")
print(df_test.columns)
print(df_train.columns)

plt.scatter(range(0, 6000), df_train["Parameter1"])
plt.show()

