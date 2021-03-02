# This script constructs a decision tree model for multi class classification using cost sensitive learning.
# Stratified cross validation is used to check the fit of the model.
# Hyperparameter tuning is used to determine the optimal parameters for the decision tree model.

# Import packages
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from sklearn.metrics import average_precision_score
# Adjust display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Loading the data
dataset = pd.read_csv(r'C:\Users\Charlie\Documents\school\Thesis\Thesis_transformed.csv')

# Preprocessing the target variable
le = LabelEncoder()
dataset.ALERT_STATUS = le.fit_transform(dataset.ALERT_STATUS)
print(dataset["ALERT_STATUS"])

# Defining the model
X = dataset.drop("ALERT_STATUS", axis=1)
print(X.shape)
y = dataset["ALERT_STATUS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#describing the model
print(dataset.info())


weights = {0: 1000, 1: 0.0001, 2: 0.0001}
# Constructing the model
model = DecisionTreeClassifier( random_state=42)
model.fit(X_train, y_train)

# Evaluation of the model
y_pred = model.predict(X_test)
con_m = confusion_matrix(y_test,y_pred)
print(con_m)
print(classification_report(y_test,y_pred))
cost = ((con_m[1,0]*5)+ (con_m[1,1]*55) + (con_m[1,2]*55) + ((con_m[2,0]+con_m[2,1]+con_m[2,2])*50))
print("total cost: " , cost)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo: %.3f" % np.mean(scores))

#hyperparameter tuning with gridsearchcv
param_grid = {"criterion":["gini", "entropy"], "max_features": ["auto", "sqrt", "log2"],
              "class_weight": [{0: 0.01, 1: 10, 2: 100}, {0: 0.1, 1: 1, 2: 10}, {0: 1, 1: 10, 2: 100}]}
cv_model = GridSearchCV(estimator=model, param_grid= param_grid, cv=5, scoring='roc_auc_ovo')
cv_model.fit(X_train,y_train)
print(cv_model.best_params_)

#resampling with SMOTE
print('Original dataset shape {}'.format(Counter(y)))
method = BorderlineSMOTE(random_state= 42)
X_res, y_res = method.fit_resample(X_train, y_train)
print('Resampled dataset shape {}'.format(Counter(y_res)))

#evaluating model with SMOTE
model.fit(X_res, y_res)
y_respred = model.predict(X_test)
con_m2 = confusion_matrix(y_test, y_respred)
print(con_m2)
print(classification_report(y_test, y_respred))
cost = ((con_m2[1,0]*5)+ (con_m2[1,1]*55) + (con_m2[1,2]*55) + ((con_m2[2,0]+con_m2[2,1]+con_m2[2,2])*50))
print("total cost: " , cost)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X_res, y_res, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo: %.3f" % np.mean(scores))
