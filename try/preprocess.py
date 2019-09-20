import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

df = pd.read_csv("../data/first_round_training_data.csv")
df_test = pd.read_csv("../data/first_round_testing_data.csv")
#parameter *10 attribute *10 Quality_label Fail Pass Good Excellent
quality_map = {"Fail":0,"Pass":1,"Good":2,'Excellent':3}
df["Quality_label"] = df["Quality_label"].map(quality_map)
df_train = df.iloc[:,0:10]
#X_train = df[:,"Parameter1":"Parameter10"]
#print(X_train.shape)
para10 = df_test.pop("Parameter10")
df_test["Parameter10"] = para10
test_group = df_test.pop("Group")
print(df_test.columns)
print(df_train.columns)

df_total = pd.concat([df_train,df_test])
print(df_train)
print(df_test)
print(df_total)
total = df_total.to_numpy()
print(total.shape)
print(total.mean(axis=0))
print(total.var(axis=0))
e = total.mean(axis=0)
s = total.std(axis=0)
total = (total-e) / s
print(total)
print(total.shape)
X_train = total[0:6000,:]
X_test  = total[6000:,:]
print(X_train.shape)
print(X_test.shape)
