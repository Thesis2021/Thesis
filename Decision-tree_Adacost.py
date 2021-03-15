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
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import make_scorer, confusion_matrix

# Adjust display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Loading the data
dataset = pd.read_csv(r'C:\Users\glenn\PycharmProjects\pythonProject1\Thesis_transformed_FLOAT.csv')

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
print(x_train.describe()
,y_train.describe())
weights = {0: 50, 1: 50, 2: 50}  # NOT FINAL

# Constructing the model
model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
model.fit(x_train, y_train)

#construct adacost

CM= np.array([[0, 0, 0], [5, 55, 55], [50, 50, 50]]) # Cost matrix

#Possible future cost calc not used atm
'''
def cost_calc(y_p,y,print_result = False):
    con_mat = confusion_matrix(y_p,y)
    cost_mat = np.multiply(con_mat,cost_matrix)
    cost = np.sum(np.multiply(con_mat,cost_matrix))/len(y)
    if print_result:
        print ("Confusion Matrix\n",con_mat)
        print ("Costs\n",cost_mat)
        print ("Total Cost = ", cost)
    else:
        return cost
'''

# Adacost                   ERROR IS HERE !!!
'''Check files for errors but none found 
x_train.to_excel("x_train.xlsx")
y_train.to_excel("y_train.xlsx")
print(x_train.isnull().values.any())
print(y_train.isnull().values.any())

'''
'''score = make_scorer(cost_calc,greater_is_better = False)'''
print(Y_train.head)

adac = AdaCost(algorithm = "SAMME.R", cost_matrix = CM, random_state = 100)
adac.fit(x_train,y_train)   #(ValueError: Input contains NaN, infinity or a value too large for dtype('float64').)
# Validated file using check null or inf. But none found. Also did manual check in Excel but non found
#Possible Solution: Use pipeline for imputing missing values, Onehotencoder and no feature selection. Hoping this would resolve the issue.
y_pred_ada = adac.predict(x_test)

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