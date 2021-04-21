# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:11:15 2020

@author: orteg
"""
import pandas as pd
import numpy as np
from boruta import BorutaPy
from utils import CatNarrow
import category_encoders as ce
import matplotlib.pyplot as plt
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from collections import Counter
import graphviz as gv

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, precision_recall_curve, \
    average_precision_score, plot_precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier


# IMPORTANT: This is the scoring function. You see as well the cost matrix.
# The scoring function gives the amount of savings you're achieving against
# no model scenario (i.e. standard investigation process for all alerts).
# For example, if your model causes the cost 20 euros and the standard investigation
# for all alerts costs 100 euros. Then the savings are calculated (100-20)/100.

def savings_investigation_score(y, y_pred):
    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    cost_matrix = np.array([[0, 0, 200],
                            [10, 60, 85],
                            [50, 75, 75]])
    cost_model = np.sum(cm * cost_matrix)
    cost_nomodel = np.sum(y == 0) * 10 + np.sum(y == 1) * 60 + np.sum(y == 2) * 85
    savings = (cost_nomodel - cost_model) / cost_nomodel
    return savings


df = pd.read_csv(r'C:\Users\glenn\PycharmProjects\pythonProject1\INT06FI_MasterThesis.csv')

cat_vars = ['CUSTOMER_TYPE.x', 'RISK_RATING.x', 'PEP_FLAG', 'OFFSHORE_FLAG',
            'BUSINESS_TYPE_DESCRIPTION', 'ADDRESS_COUNTRY_CODE',
            'NATIONALITY_COUNTRY_CODE', 'PARENT_CHILD_FLAG', 'REGISTRATION_COUNTRY_CODE',
            'DOMESTIC_PEP_CODE', 'DOMESTIC_PEP_SOURCE_CODE', 'FOREIGN_PEP_CODE',
            'FOREIGN_PEP_SOURCE_CODE', 'ADDRESS_COUNTRY_REGION',
            'ADDRESS_COUNTRY_RISK_WEIGHT', 'CUST_TYPE_ALTERNATIVE_DESCR', 'EVENT_DATE']

num_vars = ['TTL_AMOUNT_MT103_202COV_TXN',
            'TTL_COUNT_MT103_202COV_TXN', 'TTL_AMOUNT_MT103_CREDIT',
            'TTL_COUNT_MT103_CREDIT', 'TTL_AMOUNT_MT202COV_CREDIT',
            'TTL_COUNT_MT202COV_CREDIT', 'TTL_AMOUNT_MT103_DEBIT',
            'TTL_COUNT_MT103_DEBIT', 'TTL_AMOUNT_MT202COV_DEBIT',
            'TTL_COUNT_MT202COV_DEBIT', 'MAX_AMOUNT_MT103_202COV_TXN',
            'MAX_AMOUNT_MT103_202COV_THR', 'TTL_AMT_PM_MT103_202COV',
            'AVG_AMT_3M_MT103_202COV', 'PERCNT_AMT_EXCEED_PM_3M',
            'THR_AMT_EXCEED_PM_3M', 'AMOUNT_MT103_202COV_M1',
            'AMOUNT_MT103_202COV_M2', 'AMOUNT_MT103_202COV_M3',
            'TTL_CNT_PM_MT103_202COV', 'AVG_CNT_3M_MT103_202COV',
            'PERCNT_CNT_EXCEED_PM_3M', 'THR_CNT_EXCEED_PM_3M',
            'COUNT_MT103_202COV_M1', 'COUNT_MT103_202COV_M2',
            'COUNT_MT103_202COV_M3', 'CLIENT_ACC_KNOWN_FOR_MONTH']

### FEATURE ENGINEERING

# REMARK: Make sure to understand this section by checking scikit learn docs.
# WOE encoder comes from another package. You will see an intro in Baesens et al. (2015)
# CatNarrow in one function I made. It does categorization (also see Baesens et al. (2015))
# which collapses categorical levels that are less frequent than 10%.

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', RobustScaler())
                                      ]
                               )

categorical_transformer = Pipeline(steps=[('catnarrow', CatNarrow(threshold=0.10)),
                                          ('woe', ce.WOEEncoder())
                                          ]
                                   )

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_vars),
                                               ('cat', categorical_transformer, cat_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

### LABEL DEFINITION

## Multiclass Labels
cond_m1_1 = (df['ALERT_STATUS'] == 'Unsuspicious Alert')

cond_m1_2 = (df['ALERT_STATUS'] == 'Unsuspicious Case')

y_m1 = np.select([cond_m1_1, cond_m1_2], [0, 1], default=2)

## Binary label
y_b1 = np.select([cond_m1_1, cond_m1_2], [0, 0], default=1)

## Data Split

rs = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=123)
ix_tr, ix_ts = [(a, b) for a, b in rs.split(df, y_b1)][0]

X_tr_m1 = df.drop(columns='ALERT_STATUS').iloc[ix_tr]
X_ts_m1 = df.drop(columns='ALERT_STATUS').iloc[ix_ts]
y_tr_m1 = y_m1[ix_tr]
y_ts_m1 = y_m1[ix_ts]

X_tr_pp_m1 = pipe.fit_transform(X_tr_m1, y_b1[ix_tr])
X_ts_pp_m1 = pipe.transform(X_ts_m1)
X = np.concatenate((X_tr_pp_m1, X_ts_pp_m1))
y = np.concatenate((y_tr_m1, y_ts_m1))

#### MULTICLASS MODELING ####
weights = {0: 200 , 1:95, 2:125} #only takes into account missclassification cost

weights2 = {0: 200 , 1:155, 2:200} #all cost taken into account


## XGBoost ##
Xgb = xgb.XGBClassifier(objective="multi:softmax", seed=42, num_class=3, reg_lambda=0)
Xgb.fit(X_tr_pp_m1, y_m1[ix_tr], verbose=True, early_stopping_rounds=10, eval_metric='merror',
        eval_set=[(X_ts_pp_m1, y_m1[ix_ts])])
y_pred = Xgb.predict(X_ts_pp_m1)
# evaluating the model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores = cross_val_score(Xgb, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo XGBoost: %.3f" % np.mean(scores))
savings = savings_investigation_score(y_m1[ix_ts], y_pred)
print("savings XGBoost: ")
print(savings)


'''
# Parameter Tuning with GridSearchCV
param_grid = {'max_depth': [10,15,20],
              'gamma': [3, 5],
              'subsample': [0.5, 0.6, 0.7],
              'learning_rate': [0.01, 0.02, 0.03]}
cv_model = GridSearchCV(estimator=xgb, param_grid= param_grid, cv=5, scoring='roc_auc_ovo',  verbose= 2)
cv_model.fit(X_tr_pp_m1, y_m1[ix_tr])
print(cv_model.best_score_)
print(cv_model.best_params_)'''

# resampling with SMOTE
print('Original dataset shape {}'.format(Counter(y_m1)))
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_tr_pp_m1, y_tr_m1)
print('Resampled dataset shape {}'.format(Counter(y_res)))

# constructing XGBoost model with tuned parameters & SMOTE
from imblearn.pipeline import Pipeline

Xgb_tuned = xgb.XGBClassifier(objective="multi:softmax", seed=42, num_class=3, reg_lambda=0, max_depth=15, gamma=3,
                              learning_rate=0.02)
pipeline_xgb = Pipeline(steps=[('over', SMOTE()), ('XGBoost', Xgb_tuned)])
Xgb_tuned.fit(X_res, y_res, verbose=True, early_stopping_rounds=10, eval_metric='merror',
              eval_set=[(X_ts_pp_m1, y_m1[ix_ts])])
y_Xgb_respred = Xgb_tuned.predict(X_ts_pp_m1)

# Evaluating XGBoost optimal with SMOTE model
##ROC score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores = cross_val_score(pipeline_xgb, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Tuned XGBoost with SMOTE: %.3f" % np.mean(scores))
##Savings
savings = savings_investigation_score(y_m1[ix_ts], y_Xgb_respred)
print("savings Tuned XGBoost with SMOTE: ")
print(savings)


## Multiclass Random Forrest Weights 1##
rf = RandomForestClassifier(criterion= 'entropy', n_estimators= 200, max_depth= 10, n_jobs=-1, class_weight=weights)
rf.fit(X_tr_pp_m1, y_m1[ix_tr])
y_rf_pred = rf.predict(X_ts_pp_m1)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(rf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Weighted Random Forest(1): %.3f" % np.mean(scores_rf))
savings = savings_investigation_score(y_m1[ix_ts], y_rf_pred)
print("savings Weighted Random Forest(1): ")
print(savings)

print("Confusion matrix Weighted (1) Random forrest:")
print(confusion_matrix(y_ts_m1, y_rf_pred))
print("classification report Weighted (1) Random forrest")
print(classification_report(y_ts_m1, y_rf_pred))


## Multiclass Random Forrest Weights 2##
rf2 = RandomForestClassifier(criterion= 'entropy', n_estimators= 200, max_depth= 10, n_jobs=-1, class_weight=weights2)
rf2.fit(X_tr_pp_m1, y_m1[ix_tr])
y_rf_pred = rf.predict(X_ts_pp_m1)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(rf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Weighted Random Forest(2): %.3f" % np.mean(scores_rf))
savings = savings_investigation_score(y_m1[ix_ts], y_rf_pred)
print("savings Weighted Random Forest(2): ")
print(savings)

print("Confusion matrix Weighted (2) Random forrest:")
print(confusion_matrix(y_ts_m1, y_rf_pred))
print("classification report Weighted (2) Random forrest")
print(classification_report(y_ts_m1, y_rf_pred))



'''#Instantiate the grid search model
param_grid = {
              'max_depth': [15, 10, None],
              'n_estimators': [50, 100, 200]
              }
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           scoring='roc_auc_ovo',
                           cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_tr_pp_m1, y_m1[ix_tr])
print(grid_search.best_score_)
print(grid_search.best_params_)'''

'''
# Random Forest Tuned with SMOTE
rf_tuned = RandomForestClassifier(criterion='entropy', n_estimators=200, max_depth=10, n_jobs=-1, class_weight=weights)
rf_tuned.fit(X_res, y_res)
y_rf_respred = rf_tuned.predict(X_ts_pp_m1)
pipeline_rf = Pipeline(steps=[('over', SMOTE()), ('RandomForest', rf_tuned)])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(pipeline_rf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Weighted  Random Forest with SMOTE: %.3f" % np.mean(scores_rf))
savings = savings_investigation_score(y_m1[ix_ts], y_rf_respred)
print("savings Weighted  Random Forest with SMOTE: ")
print(savings)
'''

## Multiclass Decision Tree Weights 1##
dt = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=2, class_weight=weights)
dt.fit(X_tr_pp_m1, y_m1[ix_tr])
y_dt_pred = dt.predict(X_ts_pp_m1)

'''#Initiate GridsearchCV
param_grid_dt = {"max_depth": [1, 3, 5, None],
                 "criterion": ["entropy", "gini"],
                 "min_samples_split": [1, 2, 3],
                 }

grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt,
                              scoring='roc_auc_ovo',
                              cv=5, verbose=1)

grid_search_dt.fit(X_tr_pp_m1, y_m1[ix_tr])
print(grid_search_dt.best_score_)
print(grid_search_dt.best_params_)
'''
# Evaluating weighted Decision Tree Weights1
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(dt, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Weighted DecisionTree: %.3f" % np.mean(scores_rf))
savings = savings_investigation_score(y_ts_m1, y_dt_pred)
print("savings Weighted DecisionTree: ")
print(savings)

print("Confusion matrix Weighted (1) Decision Tree:")
print(confusion_matrix(y_ts_m1, y_dt_pred))
print("classification report Weighted (1) DecisionTree")
print(classification_report(y_ts_m1, y_dt_pred))


## Multiclass Decision Tree Weights 2##
dt2 = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=2, class_weight=weights2)
dt2.fit(X_tr_pp_m1, y_m1[ix_tr])
y_dt_pred = dt.predict(X_ts_pp_m1)

# Evaluating weighted Decision Tree Weights2
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(dt, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Weighted DecisionTree(2): %.3f" % np.mean(scores_rf))
savings = savings_investigation_score(y_ts_m1, y_dt_pred)
print("savings Weighted DecisionTree(2): ")
print(savings)
print("Confusion matrix Weighted (2) Decision Tree:")
print(confusion_matrix(y_ts_m1, y_dt_pred))
print("classification report Weighted (2) DecisionTree")
print(classification_report(y_ts_m1, y_dt_pred))

'''
# Optimal DecisionTree with SMOTE
dt_tuned = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=2,class_weight=weights)
dt_tuned.fit(X_res, y_res)
y_dt_respred = dt_tuned.predict(X_ts_pp_m1)
pipeline_dt = Pipeline(steps=[('over', SMOTE()), ('DecisionTree', dt_tuned)])
# Evaluating tuned DecisionTree with SMOTE
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(pipeline_dt, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Weighted DecisionTree with SMOTE: %.3f" % np.mean(scores_rf))
savings = savings_investigation_score(y_ts_m1, y_dt_respred)
print("savings Weighted DecisionTree with SMOTE: ")
print(savings)
'''


## Voting Classifier ##


Vclf = VotingClassifier(estimators=[('decisionTree', dt), ('RandomForest', rf)],
                        voting='soft')
Vclf.fit(X_tr_pp_m1, y_tr_m1)
y_Vcl_pred = Vclf.predict(X_ts_pp_m1)

# Evaluating voting classifier
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_Vclf = cross_val_score(Vclf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Voting Classifier: %.3f" % np.mean(scores_Vclf))
savings = savings_investigation_score(y_ts_m1, y_Vcl_pred)
print("savings voting Classifier: ")
print(savings)


## Voting Classifier (2) ##


Vclf = VotingClassifier(estimators=[('decisionTree', dt2), ('RandomForest', rf2)],
                        voting='soft')
Vclf.fit(X_tr_pp_m1, y_tr_m1)
y_Vcl_pred = Vclf.predict(X_ts_pp_m1)

# Evaluating voting classifier
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_Vclf = cross_val_score(Vclf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Voting Classifier (2): %.3f" % np.mean(scores_Vclf))
savings = savings_investigation_score(y_ts_m1, y_Vcl_pred)
print("savings voting Classifier (2): ")
print(savings)




'''
print("Confusion matrix Weighted RF+smote:")
print(confusion_matrix(y_ts_m1, y_rf_respred))
print("Confusion matrix Weighted DT+smote:")
print(confusion_matrix(y_ts_m1, y_dt_respred))
print("classification report Weighted RF+smote")
print(classification_report(y_ts_m1, y_rf_respred))
print("classification report Weighted DT+smote")
print(classification_report(y_ts_m1, y_dt_respred))'''