# -*- coding: utf-8 -*-
from random import seed
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import joblib


def loadData(filename):
    dataList = []
    dataSet = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip().split("\t")
            dataList.append(line)
    Features = dataList[0][1:-1]
    # Class = dataList[-1]
    for item in dataList[1:]:
        dataSet.append([float(x) for x in item[1:]])
    return Features, dataSet

def accuracy_score(x, y):
    n = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            n += 1
    accur = n / len(x)
    return accur       

# 交叉验证法
seed(2022)
Features, dataSet = loadData("pkdata.txt")
train_and_valid, test = model_selection.train_test_split(dataSet, test_size=0.5,random_state=0)
train, valid = model_selection.train_test_split(dataSet, test_size=0.1,random_state=0)

# 提取交叉验证得到的数据及标签
X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []
for item in train:
    X_train.append(item[:-1])
    y_train.append(item[-1])
for item in valid:
    X_val.append(item[:-1])
    y_val.append(item[-1])
for item in test:
    X_test.append(item[:-1])
    y_test.append(item[-1])

# 建立决策树模型
dtc = DecisionTreeClassifier(criterion="gini", min_samples_leaf=4, max_depth=10)
dtc.fit(X_train, y_train)

joblib.dump(dtc, "../../PredModel/Tree-PD.pkl")
print("Dumped.")