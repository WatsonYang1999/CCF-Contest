import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

df = pd.read_csv("../data/first_round_training_data.csv")
#parameter *10 attribute *10 Quality_label Fail Pass Good Excellent
quality_map = {"Fail":0,"Pass":1,"Good":2,'Excellent':3}
print(df)

X_train =
Y_train =