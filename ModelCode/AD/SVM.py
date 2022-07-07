import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

sns.set()

df = pd.read_csv('oasis_longitudinal.csv')
df.head()

df = df.loc[df['Visit'] == 1]  # use first visit data only because of the analysis we're doing
df = df.reset_index(drop=True)  # reset index after filtering first visit data
df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])  # M/F column
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])  # Target variable
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])  # Target variable
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)  # Drop unnecessary columns

# Check missing values by each column
pd.isnull(df).sum()
# The column, SES has 8 missing values

df_dropna = df.dropna(axis=0, how='any')
pd.isnull(df_dropna).sum()

df_dropna['Group'].value_counts()

# Draw scatter plot between EDUC and SES
x = df['EDUC']
y = df['SES']

ses_not_null_index = y[~y.isnull()].index
x = x[ses_not_null_index]
y = y[ses_not_null_index]

df.groupby(['EDUC'])['SES'].median()

df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)

# I confirm there're no more missing values and all the 150 data were used.
pd.isnull(df['SES']).value_counts()

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

# Dataset with imputation
Y = df['Group'].values  # Target for the model
X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]  # Features we use

# splitting into three sets
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, random_state=0)

# Feature scaling
scaler = MinMaxScaler().fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# Dataset after dropping missing value rows
Y = df_dropna['Group'].values  # Target for the model
X = df_dropna[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]  # Features we use

# splitting into three sets
X_trainval_dna, X_test_dna, Y_trainval_dna, Y_test_dna = train_test_split(X, Y, random_state=0, test_size=0.2)

# Feature scaling
scaler = MinMaxScaler().fit(X_trainval_dna)
X_trainval_scaled_dna = scaler.transform(X_trainval_dna)
X_test_scaled_dna = scaler.transform(X_test_dna)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc, precision_score

# SVM
acc = []
best_score = 0
kfolds = 5

for c_paramter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:  # iterate over the values we need to try for the parameter C
    for gamma_paramter in [0.001, 0.01, 0.1, 1, 10, 100,
                           1000]:  # iterate over the values we need to try for the parameter gamma
        for k_parameter in ['rbf', 'linear', 'poly',
                            'sigmoid']:  # iterate over the values we need to try for the kernel parameter
            svmModel = SVC(kernel=k_parameter, C=c_paramter, gamma=gamma_paramter)  # define the model
            # perform cross-validation
            scores = cross_val_score(svmModel, X_trainval_scaled, Y_trainval, cv=kfolds, scoring='accuracy')
            # the training set will be split internally into training and cross validation

            # compute mean cross-validation accuracy
            score = np.mean(scores)
            # if we got a better score, store the score and parameters
            if score > best_score:
                best_score = score  # store the score
                best_parameter_c = c_paramter  # store the parameter c
                best_parameter_gamma = gamma_paramter  # store the parameter gamma
                best_parameter_k = k_parameter

# rebuild a model with best parameters to get score
SelectedSVMmodel = SVC(C=best_parameter_c, gamma=best_parameter_gamma, kernel=best_parameter_k).fit(X_trainval_scaled,
                                                                                                    Y_trainval)

test_score = SelectedSVMmodel.score(X_test_scaled, Y_test)
PredictedOutput = SelectedSVMmodel.predict(X_test_scaled)
test_precision = precision_score(Y_test, PredictedOutput, pos_label=1)
test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
test_auc = auc(fpr, tpr)

print("Best accuracy on cross validation set is:", best_score)
print("Best parameter for c is: ", best_parameter_c)
print("Best parameter for gamma is: ", best_parameter_gamma)
print("Best parameter for kernel is: ", best_parameter_k)
print("Test accuracy with the best parameters is", test_score)
print("Test precision with the best parameters is", test_precision)
print("Test recall with the best parameters is", test_recall)
print("Test recall with the best parameter is", test_auc)

m = 'SVM'
acc.append([m, test_score, test_recall, test_precision, test_auc, fpr, tpr, thresholds])

# Write Model
joblib.dump(SelectedSVMmodel, "../../PredModel/SVM-AD.pkl")
print("Dumped.")

