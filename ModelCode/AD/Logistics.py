import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_curve, auc

import joblib

warnings.filterwarnings("ignore")

# Set Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(suppress=True)

# 读入数据
df = pd.read_csv('oasis_longitudinal.csv')
df = df.loc[df['Visit'] == 1]
df = df.reset_index(drop=True)
df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)
df_dropna = df.dropna(axis=0, how='any')




# 首先尝试选择删去这些含有缺失值的行
df_dropna = df.dropna(axis=0, how='any')

# 删除缺失值后的数据划分与训练
Y = df_dropna['Group'].values  # Target for the model
X = df_dropna[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]  # 不采用CDR这一特征


# 分为训练集和测试集
X_trainval_dna, X_test_dna, Y_trainval_dna, Y_test_dna = train_test_split(X, Y, random_state=0)
# 特征标准化
scaler_1 = MinMaxScaler().fit(X_trainval_dna)
X_trainval_scaled_dna = scaler_1.transform(X_trainval_dna)
X_test_scaled_dna = scaler_1.transform(X_test_dna)

acc = []  # list to store all performance metric
best_score = 0
kfolds = 5
for c in [0.001, 0.1, 1, 10, 100]:
    logRegModel = LogisticRegression(C=c)
    # 交叉验证
    scores = cross_val_score(logRegModel, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')
    # 计算交叉验证准确性的均值
    score = np.mean(scores)
    # 寻找最佳参数和得分
    if score > best_score:
        best_score = score
        best_parameters = c
# 在综合训练和验证集上重建一个模型
SelectedLogRegModel_1 = LogisticRegression(C=best_parameters).fit(X_trainval_scaled_dna, Y_trainval_dna)
test_score = SelectedLogRegModel_1.score(X_test_scaled_dna, Y_test_dna)
PredictedOutput = SelectedLogRegModel_1.predict(X_test_scaled_dna)
test_recall = recall_score(Y_test_dna, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test_dna, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)
print("Best accuracy on validation set is:", best_score)
print("Best parameter for regularization (C) is: ", best_parameters)
print("Test accuracy with best C parameter is：", test_score)
print("Test recall with the best C parameter is：", test_recall)
print("Test AUC with the best C parameter is：", test_auc)
m = 'Logistic Regression (dropna)'
acc.append([m, test_score, test_recall, test_auc, fpr, tpr, thresholds])

"""Best accuracy on validation set is: 0.725974025974026
Best parameter for regularization (C) is:  10
Test accuracy with best C parameter is： 0.8055555555555556
Test recall with the best C parameter is： 0.9411764705882353
Test AUC with the best C parameter is： 0.8126934984520124"""

# 补充计算：TN TP FN FP
cm = confusion_matrix(Y_test_dna, PredictedOutput)
# label the confusion matrix
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
# set size of the plot
plt.figure(figsize=(8, 5))
# plot a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
TN = cm[0, 0]
# True Positives are denoted by 'TP'
# Actual '1' values which are classified correctly
TP = cm[1, 1]
# False Negatives are denoted by 'FN'
# Actual '1' values which are classified wrongly as '0'
FN = cm[1, 0]
# False Positives are denoted by 'FP'
# Actual 'O' values which are classified wrongly as '1'
FP = cm[0, 1]
result = classification_report(Y_test_dna, PredictedOutput)

pr = TP/(TP+FP)
print(pr)
# print the result
print(result)
#              precision    recall  f1-score   support

#           0       0.93      0.68      0.79        19
#           1       0.73      0.94      0.82        17
#
#    accuracy                           0.81        36
#   macro avg       0.83      0.81      0.80        36
# weighted avg       0.83      0.81      0.80        36


joblib.dump(SelectedLogRegModel_1, "../../PredModel/Logistics-AD.pkl")
print("Dumped.")
