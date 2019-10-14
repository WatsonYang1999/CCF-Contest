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


df_total = pd.concat([df_train,df_test])

total = df_total.to_numpy()

e = total.mean(axis=0)
s = total.std(axis=0)
total = (total-e) / s

X_train = total[0:6000, :]
size = 6000
print(Y_train)
for i in range(0):
    X_train = np.concatenate([X_train,X_train])
    Y_train = np.concatenate([Y_train,Y_train],axis=0)
    size*=2
X_test = total[6000:, :]
print(Y_train.shape)
print(X_train.shape)
# abnormal_index = []
# for j in range(10):
#     vector_c = X_train[:,j]
#     max_index = np.argmax(vector_c)
#     X_train = np.delete(X_train,max_index,0)
#     vector_c = X_train[:, j]
#     min_index = np.argmin(vector_c)
#     X_train = np.delete(X_train,min_index,0)
#
# print(X_train.shape)

clf = LogisticRegression(solver='lbfgs',multi_class='multinomial').fit(X_train,Y_train)
clf = tree.DecisionTreeClassifier()
#clf = MLPClassifier(solver = "lbfgs",alpha = 1e-5,hidden_layer_sizes=(10,10,10),random_state=1)
clf = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
train_pred = clf.predict(X_train)
hit = 0
print(Y_train.shape)
print(Y_pred.shape)
for index in range(Y_pred.size):
    if train_pred[index] == Y_train[index]:
        hit += 1
print("hit %d Accuracy : %f", hit, hit/6000)

print(Y_train[0:100])
print(train_pred[0:100])

stat = np.zeros([120,4],dtype=int)
label_map = {0:"Fail",1:"Pass", 2:"Good",3:"Excellent"}
#print(df_train)
for i,label in enumerate(Y_pred):
    group = test_group[i]
    stat[group][label] += 1

stat = stat.astype(dtype=float)


for index, row in enumerate(stat):
    s = row.sum()

    stat[index] /= s

df_submit = pd.DataFrame(stat,
                         columns=["Excellent ratio","Good ratio","Pass ratio","Fail ratio"])
df_submit.to_csv("submission.csv",index_label="Group")







