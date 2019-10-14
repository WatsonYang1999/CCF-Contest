from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import pandas as pd
arr = np.array([[1,2,3],[4,5,6]])
df = pd.DataFrame(arr)
print(df)
std = df.std(axis=0)
print(std)
df = df/2
print(df)