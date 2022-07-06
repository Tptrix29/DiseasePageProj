# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:15:10 2022

@author: lenovo
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc

# 处理数据，将一些特征分类转换为数字，删除不需要的列
sns.set()
df = pd.read_csv('oasis_longitudinal.csv')
df.head()
df = df.loc[df['Visit'] == 1]
df = df.reset_index(drop=True)
df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])  # M/F column;将性别转变为0/1值
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])  # 将转换后的非痴呆患者转变为痴呆患者
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])  # 将痴呆和非痴呆转换为0/1值
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)  # 删去不重要的列

# 检查缺失值
pd.isnull(df).sum()
# SES列有8个缺失值
# 将八个缺失确实值所在的行删去
df_dropna = df.dropna(axis=0, how='any')
pd.isnull(df_dropna).sum()
df_dropna['Group'].value_counts()

# 删去缺失值后的数据集
Y = df_dropna['Group'].values  # Target for the model
X = df_dropna[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]  # Features we use

# 将数据集划分为训练集、测试集
X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    X, Y, random_state=0, test_size=0.2)

# 标准化归一化
scaler = MinMaxScaler().fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

acc = []  # 储存模型表现

best_score = 0
kfolds = 5  # 五折交叉验证

for M in range(2, 15, 2):  # combines M trees
    for lr in [0.0001, 0.001, 0.01, 0.1, 1]:
        # train the model
        boostModel = AdaBoostClassifier(n_estimators=M, learning_rate=lr, random_state=0)

        # perform cross-validation
        scores = cross_val_score(boostModel, X_trainval_scaled, Y_trainval, cv=kfolds, scoring='accuracy')

        # compute mean cross-validation accuracy
        score = np.mean(scores)

        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_M = M
            best_lr = lr

# Rebuild a model on the combined training and validation set
SelectedBoostModel = AdaBoostClassifier(n_estimators=M, learning_rate=lr, random_state=0).fit(X_trainval_scaled,
                                                                                              Y_trainval)

import joblib


# 保存模型
# def save_model(model, filepath):
#     # 后缀一般用pkl
#     joblib.dump(model, filename=filepath)


# save_model(SelectedBoostModel, 'AdaBoost-AD.pkl')

PredictedOutput = SelectedBoostModel.predict(X_test_scaled)
test_score = SelectedBoostModel.score(X_test_scaled, Y_test)
test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)
print("Best accuracy on validation set is:", best_score)
print("Best parameter of M is: ", best_M)
print("best parameter of LR is: ", best_lr)
print("Test accuracy with the best parameter is", test_score)
print("Test recall with the best parameters is:", test_recall)
print("Test AUC with the best parameters is:", test_auc)

m = 'AdaBoost'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])

print("Feature importance: ")
print(np.array([X.columns.values.tolist(), list(SelectedBoostModel.feature_importances_)]).T)

joblib.dump(SelectedBoostModel, "../../PredModel/AdaBoost-AD.pkl")
print("Dumped.")