# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:29:55 2020

@author: skambou
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.stats.contingency import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, f1_score

df = pd.read_csv('D:\\Data Science\\ML\\Project\\Classification\\bank.csv')

df.shape
df.columns
df.describe()
df.isnull().sum()
df.info()

df2 = df
# Encoding



df2.job.unique()

df2.job = df2.job.map({
                                'unemployed':0, 'services':1, 
                                'management':2, 'blue-collar':3,
                                'self-employed':4, 'technician':5, 
                                'entrepreneur':6, 'admin.':7, 
                                'student':8,'housemaid':9, 
                                'retired':10, 'unknown':11
                                })

df2.marital.unique()

df2.marital = df2.marital.map({'married': 0, 'single': 1, 'divorced':2})


df2.education.unique()

df2.education = df2.education.map({'primary':0, 'secondary':1, 'tertiary':2, 'unknown':3})


df2.default.unique()

df2.default = df2.default.map({'no':0, 'yes':1})


df2.housing.unique()

df2.housing = df2.housing.map({'no':0, 'yes':1})


df2.loan.unique()

df2.loan = df2.loan.map({'no':0, 'yes':1})


df2.contact.unique()

df2.contact = df2.contact.map({'unknown':0, 'cellular':1, 'telephone':2})


df2.month.unique()

df2.month = df2.month.map({'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                           'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12})


df2.poutcome.unique()

df2.poutcome = df2.poutcome.map({'failure':0, 'success':1, 'unknown':2, 'other':3})


df2.subscribe.unique()

df2.subscribe = df2.subscribe.map({'no':0, 'yes':1})

df2.isnull().sum()

""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""
The classification goal is to predict if
 the client will subscribe (yes/no) a term deposit.
"""""""""""""""""""""""""""""""""

X = df2.iloc[:,:-1]
y = df2.iloc[:,-1]

# split in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)


# scale x
sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns=X_train.columns)

#import linear model Ordinary Least Square
import statsmodels.api as sm
#X_train = X_train
#X2_test = X_test
y_train = np.array(y_train)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
ols = sm.OLS(y_train,X_train)
lr = ols.fit()

print(lr.summary())


X_train.drop(['age', 'default', 'balance', 'day', 'month','campaign', 'poutcome'], axis=1, inplace=True)
X_test.drop(['age', 'default', 'balance', 'day', 'month','campaign', 'poutcome'], axis=1, inplace=True)

ols = sm.OLS(y_train,X_train)
lr = ols.fit()
print(lr.summary())


"""
Logistic Regression Analysis
"""
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train, y_train)

y_pred_log = log.predict(X_test)


#performance measures
cm = confusion_matrix(y_test, y_pred_log)
accuracy = accuracy_score(y_test, y_pred_log)
precision = precision_score(y_test, y_pred_log)
sensitivity = recall_score(y_test, y_pred_log)
print(classification_report(y_test, y_pred_log))
print('Accuracy:', accuracy)
print('Precision:', precision)


# Confusion Matrix

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Greens')
plt.title('Confusion Matrix/n')
plt.show()


'''
KNN ANALYSIS

'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'n_neighbors': range(1,10), 'p': range(1,10)}]
knn = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=4)
knn.fit(X_train, y_train)
print("Best number of neighbors found {}:".format(knn.best_params_))

#will use K = 6
knn = KNeighborsClassifier(n_neighbors=6, p=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))
print('Accuracy:', accuracy_knn)

# Confusion Matrix KNN

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm_knn.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm_knn.flatten()/np.sum(cm_knn)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_knn, annot=labels, fmt='', cmap='Greens')
plt.title('Confusion Matrix/n')
plt.show()


"""
ADABOOST ANALYSIS
"""
from sklearn.ensemble import AdaBoostClassifier

tuned_parameters = [{'n_estimators': range(10,110,10), 'random_state': range(0,60,10)}]
ada = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=4)
ada.fit(X_train, y_train)
print("Best number of estimators found {}:".format(ada.best_params_))

#best # estimators is 30

ada = AdaBoostClassifier(n_estimators=30, random_state = 0)
ada.fit(X_train, y_train)
ada.score(X_test, y_test)

y_pred_ada = ada.predict(X_test)

cm_ada = confusion_matrix(y_test, y_pred_ada)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(classification_report(y_test, y_pred_ada))
print('Accuracy:', accuracy_ada)

# Confusion Matrix Adaboost

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm_ada.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm_ada.flatten()/np.sum(cm_ada)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_ada, annot=labels, fmt='', cmap='Greens')
plt.title('Confusion Matrix/n')
plt.show()


"""
RANDOM FOREST ANALYSIS
"""

from sklearn.ensemble import RandomForestClassifier
tuned_parameters = [{'n_estimators': [10,110,10], 
                     'max_depth': [1,11], 
                     'random_state':[0,60,10]}]
rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv = 4)
rf.fit(X_train, y_train)
print("Best estimators found {}:".format(rf.best_params_))



rf = RandomForestClassifier(n_estimators=110, max_depth=11, random_state=60)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

y_pred_rf = rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf))
print('Accuracy:', accuracy_rf)

# Confusion Matrix Random Forest

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm_rf.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm_rf.flatten()/np.sum(cm_rf)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_rf, annot=labels, fmt='', cmap='Greens')
plt.title('Confusion Matrix/n')
plt.show()


"""
SVM Analysis
"""

from sklearn.svm import SVC

tuned_parameters = [{'kernel': ['linear', 'poly', 'rbf'],}]
svm = GridSearchCV(SVC(), tuned_parameters, cv = 4)
svm.fit(X_train, y_train)
print("Best estimators found {}:".format(svm.best_params_))

tuned_parameters = [{'degree' : [2,3]}]
svm = GridSearchCV(SVC(), tuned_parameters, cv = 4)
svm.fit(X_train, y_train)
print("Best estimators found {}:".format(svm.best_params_))

tuned_parameters = [{'gamma' : [0.01, 0.1,1,10],
                     'C': [0.01, 0.1,1,10]}]
svm = GridSearchCV(SVC(), tuned_parameters, cv = 4)
svm.fit(X_train, y_train)
print("Best estimators found {}:".format(svm.best_params_))


svm = SVC(kernel='rbf', degree = 2, C=1, gamma=1)
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

y_pred_svm = svm.predict(X_test)

cm_svm = confusion_matrix(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(classification_report(y_test, y_pred_svm))
print('Accuracy:', accuracy_svm)

# Confusion Matrix Random Forest

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm_svm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm_svm.flatten()/np.sum(cm_svm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_svm, annot=labels, fmt='', cmap='Greens')
plt.title('Confusion Matrix/n')
plt.show()


print('Accuracy for Logistic Regression: ', accuracy*100)
print('Accuracy for KNN: ', accuracy_knn*100)
print('Accuracy for Adaboost: ', accuracy_ada*100)
print('Accuracy for RandomForest:', accuracy_rf*100)
print('Accuracy for SVC: ', accuracy_svm*100)
print('')
print('Confusion Matrix for Logistic Regression :\n', cm_rf)
print('Confusion Matrix for KNN :\n', cm_knn)
print('Confusion Matrix Adaboost :\n', cm_ada)
print('Confusion Matrix RandomForest :\n', cm_rf)
print('Confusion Matrix SVC :\n', cm_svm)


""" Based on the results, we can say that Random Forest is the current best model for 
this dataset analysis"""



#Kfold cross validation

#preprocessing
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Perform 4-fold cross validation
scores = cross_val_score(rf, X_scaled, y, cv=4)
print ('Cross-validated scores:', scores)
