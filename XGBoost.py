import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter

# Loading the data
dataset = pd.read_csv(r'C:\Users\Charlie\Documents\school\Thesis\Thesis_transformed.csv')

# Preprocessing the target variable
le = LabelEncoder()
dataset.ALERT_STATUS = le.fit_transform(dataset.ALERT_STATUS)

# Defining the model
X = dataset.drop("ALERT_STATUS", axis=1)
print(X.shape)
y = dataset["ALERT_STATUS"]
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

'''following part is marked as comments to speed up de computing process once the tuning is done'''

# Constructing the model
# xgb = xgb.XGBClassifier(objective="multi:softmax", seed=42, num_class=3 )
# xgb.fit(X_train,y_train, verbose=True, early_stopping_rounds= 10, eval_metric='merror', eval_set=[(X_test,y_test)])
# y_pred = xgb.predict(X_test)
#evaluating the model
# plot_confusion_matrix(xgb, X_test, y_test, values_format='d', display_labels = ["unSuspicious Alert", "Unsuspicious Case", "Suspicious Activity Report"])
# plt.show()
# con_m = confusion_matrix(y_test, y_pred)
# print(con_m)
# print(classification_report(y_test, y_pred))
# cost = ((con_m[1, 0]*5) + (con_m[1, 1]*55) + (con_m[1, 2]*55) + ((con_m[2, 0]+con_m[2, 1]+con_m[2, 2])*50))
# print("total cost: ", cost)

# Parameter Tuning with GridSearchCV
#Round1
#param_grid = {'max_depth': [3,4,5],
#              'learning_rate': [0.1,0.01,0.05],
#              'gamma': [0, 0.25, 1.0],
#              'reg_lambda': [0, 1.0, 10.0],
#              'subsample': [0.5, 1]}
#cv_model = GridSearchCV(estimator=xgb, param_grid= param_grid, cv=5, scoring='roc_auc_ovo')
#cv_model.fit(X_train, y_train)
#print(cv_model.best_params_)
#Round2
#param_grid2 = {'max_depth': [1,2,3],
#              'learning_rate': [0.1,0.2, 0.3],
#              'gamma': [0],
#              'reg_lambda': [1]}
#cv_model = GridSearchCV(estimator=xgb, param_grid= param_grid2, cv=5, scoring='roc_auc_ovo')
#cv_model.fit(X_train, y_train)
#print(cv_model.best_params_)

# Constructing the model with optimal parameters
xgb_optimal = xgb.XGBClassifier(objective="multi:softmax", seed=42, num_class=3, gamma=0, learning_rate=0.1, max_depth=3, reg_lambda=1)
xgb_optimal.fit(X_train,y_train, verbose=True, early_stopping_rounds= 10, eval_metric='merror', eval_set=[(X_test, y_test)])
y_pred = xgb_optimal.predict(X_test)
#evaluating the model
plot_confusion_matrix(xgb_optimal, X_test, y_test, values_format='d', display_labels = ["unSuspicious Alert", "Unsuspicious Case", "Suspicious Activity Report"])
plt.show()
con_m = confusion_matrix(y_test, y_pred)
print(con_m)
print(classification_report(y_test, y_pred))
cost = ((con_m[1, 0]*5) + (con_m[1, 1]*55) + (con_m[1, 2]*55) + ((con_m[2, 0]+con_m[2, 1]+con_m[2, 2])*50))
print("total cost: ", cost)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(xgb_optimal, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo: %.3f" % np.mean(scores))

#resampling with SMOTE
print('Original dataset shape {}'.format(Counter(y)))
method = BorderlineSMOTE(random_state= 42)
X_res, y_res = method.fit_resample(X_train, y_train)
print('Resampled dataset shape {}'.format(Counter(y_res)))

#constructing model with optimal parameters & SMOTE
xgb_optimal.fit(X_res,y_res, verbose=True, early_stopping_rounds=10, eval_metric='merror', eval_set=[(X_test,y_test)])
y_respred= xgb_optimal.predict(X_test)
con_m2 = confusion_matrix(y_test, y_respred)
print(con_m2)
print(classification_report(y_test, y_respred))
cost = ((con_m2[1,0]*5)+ (con_m2[1,1]*55) + (con_m2[1,2]*55) + ((con_m2[2,0]+con_m2[2,1]+con_m2[2,2])*50))
print("total cost: " , cost)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(xgb_optimal, X_res, y_res, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo: %.3f" % np.mean(scores))