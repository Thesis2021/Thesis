# This script constructs a decision tree model for multi class classification using cost sensitive learning.
# Stratified cross validation is used to check the fit of the model.
# Hyperparameter tuning is used to determine the optimal parameters for the decision tree model.

# Import packages
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Adjust display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Loading the data
dataset = pd.read_csv('Thesis_transformed2.csv')

# Preprocessing the target variable
#le = LabelEncoder()
#dataset.ALERT_STATUS = le.fit_transform(dataset.ALERT_STATUS)
#print(dataset["ALERT_STATUS"])

# Defining the model
x = dataset.drop("ALERT_STATUS", axis=1)
print(x)
print(x.shape)
y = dataset["ALERT_STATUS"]
print(y)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

weights = {0: 50, 1: 50, 2: 50}  # NOT FINAL

# Constructing the model
model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
model.fit(x_train, y_train)

# Evaluation of the model
y_pred = model.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, x, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("Mean ROC AUC: %.3f" % np.mean(scores))

# Hyperparameter tuning for the decision tree parameters
parameterGrid = {"max_depth": list(np.linspace(1, 5, 10, 15, dtype=int)) + [None],
                 "criterion": ["entropy", "gini"],
                 "min_samples_split": [2, 3, 4, 5, 6],
                 "ccp_alpha": list(np.linspace(0, 1, 50, dtype=float))}
# print(parameterGrid)

candidateModel = RandomizedSearchCV(model, param_distributions=parameterGrid, scoring="f1_macro", cv=5, n_iter=10, random_state=42)
candidateModel.fit(x_train, y_train)
print(candidateModel.best_params_)

# Constructing a decision tree with the acquired optimal parameters
# optimalModel = DecisionTreeClassifier(candidateModel.best_params_)
optimalModel = DecisionTreeClassifier(min_samples_split=3, max_depth=3, criterion="entropy", ccp_alpha=0.08163265306122448, random_state=42)
optimalModel.fit(x_train, y_train)

optimal_y_pred = optimalModel.predict(x_test)
print(confusion_matrix(y_test, optimal_y_pred))
print(classification_report(y_test, optimal_y_pred))

scores = cross_val_score(optimalModel, x, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("Mean ROC AUC: %.3f" % np.mean(scores))