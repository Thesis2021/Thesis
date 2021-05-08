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
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, precision_recall_curve, average_precision_score, plot_precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression



# IMPORTANT: This is the scoring function. You see as well the cost matrix.
# The scoring function gives the amount of savings you're achieving against 
# no model scenario (i.e. standard investigation process for all alerts).
# For example, if your model causes the cost 20 euros and the standard investigation 
# for all alerts costs 100 euros. Then the savings are calculated (100-20)/100.

def savings_investigation_score(y, y_pred):
  cm = confusion_matrix(y, y_pred, labels=[0,1,2])
  cost_matrix = np.array([[0,0,200],
                          [10,60,85],
                          [50,75,75]])
  cost_model = np.sum(cm*cost_matrix)
  cost_nomodel = np.sum(y==0)*10 + np.sum(y==1)*60 + np.sum(y==2)*85
  savings = (cost_nomodel - cost_model)/cost_nomodel
  return savings

df = pd.read_csv(r'C:\Users\Charlie\Documents\school\Thesis\INT06FI_MasterThesis.csv')

cat_vars = ['CUSTOMER_TYPE.x','RISK_RATING.x','PEP_FLAG','OFFSHORE_FLAG',
            'BUSINESS_TYPE_DESCRIPTION','ADDRESS_COUNTRY_CODE',
            'NATIONALITY_COUNTRY_CODE','PARENT_CHILD_FLAG','REGISTRATION_COUNTRY_CODE',
            'DOMESTIC_PEP_CODE','DOMESTIC_PEP_SOURCE_CODE','FOREIGN_PEP_CODE',
            'FOREIGN_PEP_SOURCE_CODE','ADDRESS_COUNTRY_REGION',
            'ADDRESS_COUNTRY_RISK_WEIGHT','CUST_TYPE_ALTERNATIVE_DESCR','EVENT_DATE']

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

numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', RobustScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('catnarrow', CatNarrow(threshold = 0.10)),
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

y_m1 = np.select([cond_m1_1, cond_m1_2], [0,1], default = 2)

## Binary label
y_b1 = np.select([cond_m1_1, cond_m1_2], [0,0], default = 1)

## Data Split

rs = StratifiedShuffleSplit(n_splits = 1, test_size = 0.30, random_state = 123)
ix_tr, ix_ts = [(a, b) for a, b in rs.split(df, y_b1)][0]

X_tr_m1 = df.drop(columns = 'ALERT_STATUS').iloc[ix_tr]
X_ts_m1 = df.drop(columns = 'ALERT_STATUS').iloc[ix_ts] 
y_tr_m1 = y_m1[ix_tr]
y_ts_m1 = y_m1[ix_ts]

X_tr_pp_m1 = pipe.fit_transform(X_tr_m1, y_b1[ix_tr])
X_ts_pp_m1 = pipe.transform(X_ts_m1)
X = np.concatenate((X_tr_pp_m1, X_ts_pp_m1))
y = np.concatenate((y_tr_m1, y_ts_m1))

#resampling with SMOTE
print('Original dataset shape {}'.format(Counter(y_m1)))
smote = SMOTE(random_state= 42)
X_res, y_res = smote.fit_resample(X_tr_pp_m1, y_tr_m1)
print('Resampled dataset shape {}'.format(Counter(y_res)))
from imblearn.pipeline import Pipeline

#### MULTICLASS MODELING ####

## Logistic Regression ##
lr = LogisticRegression()
lr.fit(X_tr_pp_m1, y_m1[ix_tr])
y_lr_pred= lr.predict(X_ts_pp_m1)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_lr = cross_val_score(lr, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo lr: %.3f" % np.mean(scores_lr))
savings = savings_investigation_score(y_m1[ix_ts], y_lr_pred)
print("savings  lr: ")
print(savings)

print("Confusion matrix lr:")
print(confusion_matrix(y_ts_m1, y_lr_pred))
print("classification report lr")
print(classification_report(y_ts_m1, y_lr_pred))

# Parameter Tuning logistic regression with GridSearchCV
param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'multi_class': ["ovr", "multinomial"]
             }
cv_model = GridSearchCV(estimator=lr, param_grid= param_grid, cv=5, scoring='roc_auc_ovo',  verbose= 2)
cv_model.fit(X_tr_pp_m1, y_m1[ix_tr])
print(cv_model.best_score_)
print(cv_model.best_params_)

# evaluation tuned model
lr_tuned = LogisticRegression(multi_class='multinomial', penalty='l2', solver='newton-cg')
lr_tuned.fit(X_tr_pp_m1, y_tr_m1)
y_lr_respred = lr_tuned.predict(X_ts_pp_m1)

pipeline_lr = Pipeline(steps= [('over', SMOTE()), ('LogisticRegression', lr_tuned)])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_lr = cross_val_score(pipeline_lr, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo tuned lr with SMOTE: %.3f" % np.mean(scores_lr))
savings = savings_investigation_score(y_m1[ix_ts], y_lr_respred)
print("savings tuned lr with SMOTE: ")
print(savings)

print("classification report tuned lr with SMOTE")
print(classification_report(y_ts_m1, y_lr_respred))


## XGBoost ##
Xgb = xgb.XGBClassifier(objective="multi:softmax", seed=42, num_class=3, reg_lambda=0)
Xgb.fit(X_tr_pp_m1, y_m1[ix_tr], verbose=True, early_stopping_rounds=10, eval_metric='merror',
        eval_set=[(X_ts_pp_m1, y_m1[ix_ts])])
y_pred = Xgb.predict(X_ts_pp_m1)
#evaluating the model
print('classification report XGBoost')
print(classification_report(y_ts_m1, y_pred))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores = cross_val_score(Xgb, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo XGBoost: %.3f" % np.mean(scores))
savings = savings_investigation_score(y_m1[ix_ts], y_pred)
print("savings XGBoost: ")
print(savings)

# Parameter Tuning with GridSearchCV
param_grid = {'max_depth': [10, 15, 20],
              'gamma': [3, 4, 5],
              'subsample': [0.3, 0.5, 0.75],
              'learning_rate': [0.02, 0.03, 0.04],
              'reg_lambda': [0, 1]}
cv_model = GridSearchCV(estimator=Xgb, param_grid= param_grid, cv=5, scoring='roc_auc_ovo',  verbose= 2)
cv_model.fit(X_tr_pp_m1, y_m1[ix_tr])
print(cv_model.best_score_)
print(cv_model.best_params_)

#constructing XGBoost model with tuned parameters & SMOTE
Xgb_tuned = xgb.XGBClassifier(objective="multi:softmax", seed=42, num_class=3, reg_lambda=0, max_depth=15, gamma=4,
                              learning_rate=0.03, subsample=0.5)
Xgb_tuned.fit(X_res,y_res, verbose=True, early_stopping_rounds=10, eval_metric='merror',
                eval_set=[(X_ts_pp_m1,y_m1[ix_ts])])
y_Xgb_respred= Xgb_tuned.predict(X_ts_pp_m1)
pipeline_xgb = Pipeline(steps= [('over', SMOTE()), ('XGBoost', Xgb_tuned)])
#Evaluating XGBoost tuned with SMOTE model
print('classification report XGBoost tuned with SMOTE ')
print(classification_report(y_ts_m1, y_Xgb_respred))
 ##ROC score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores = cross_val_score(pipeline_xgb, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Tuned XGBoost with SMOTE : %.3f" % np.mean(scores))
 ##Savings
savings = savings_investigation_score(y_m1[ix_ts], y_Xgb_respred)
print("savings Tuned XGBoost with SMOTE : ")
print(savings)



## Multiclass Random Forrest ##
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X_tr_pp_m1, y_m1[ix_tr])
y_rf_pred = rf.predict(X_ts_pp_m1)
#Evaluation Random Forrest
print('classification report Random Forrest')
print(classification_report(y_ts_m1, y_rf_pred))
 ##ROC score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(rf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Random Forest: %.3f" % np.mean(scores_rf))
 ##Savings
savings = savings_investigation_score(y_m1[ix_ts], y_rf_pred)
print("savings Random Forest: ")
print(savings)

#Instantiate the grid search model
param_grid = {
              'criterion': ['entropy', 'gini'],
              'max_depth': [15, 10, 5],
              'n_estimators': [200, 250, 300]
              }
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           scoring='roc_auc_ovo',
                           cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_tr_pp_m1, y_m1[ix_tr])
print(grid_search.best_score_)
print(grid_search.best_params_)

# Random Forest Tuned with SMOTE
rf_tuned = RandomForestClassifier(criterion= 'entropy', n_estimators= 250, max_depth= 10, n_jobs=-1)
rf_tuned.fit(X_res, y_res)
y_rf_respred = rf_tuned.predict(X_ts_pp_m1)
pipeline_rf = Pipeline(steps= [('over', SMOTE()), ('RandomForest', rf_tuned)])
# Evaluating Random Forest tuned with SMOTE
print('classification report Random forest tuned with SMOTE')
print(classification_report(y_ts_m1, y_rf_respred))
 ##ROC score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(pipeline_rf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Tuned Random Forest with SMOTE: %.3f" % np.mean(scores_rf))
 ##Savings
savings = savings_investigation_score(y_m1[ix_ts], y_rf_respred)
print("savings Tuned Random Forest with SMOTE: ")
print(savings)



## Multiclass Decision Tree ##
dt = DecisionTreeClassifier()
dt.fit(X_tr_pp_m1, y_tr_m1)
y_dt_pred = dt.predict(X_ts_pp_m1)
# Evaluating Decision Tree
print('classification report Decision Tree')
print(classification_report(y_ts_m1, y_dt_pred))
 ##ROC score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(dt, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo DecisionTree: %.3f" % np.mean(scores_rf))
 ##Savings
savings = savings_investigation_score(y_ts_m1, y_dt_pred)
print("savings DecisionTree: ")
print(savings)

#Initiate GridsearchCV
param_grid_dt = {"max_depth": [1, 3, 5],
                 "criterion": ["entropy", "gini"],
                 "min_samples_split": [2, 3],
                 }

grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt,
                              scoring='roc_auc_ovo',
                              cv=5, verbose=1)

grid_search_dt.fit(X_tr_pp_m1, y_tr_m1)
print(grid_search_dt.best_score_)
print(grid_search_dt.best_params_)

# Tuned DecisionTree with SMOTE
dt_tuned = DecisionTreeClassifier(criterion='gini', max_depth= 3, min_samples_split= 2)
dt_tuned.fit(X_res, y_res)
y_dt_respred = dt_tuned.predict(X_ts_pp_m1)
pipeline_dt = Pipeline(steps= [('over', SMOTE()), ('DecisionTree', dt_tuned)])
#Evaluating tuned DecisionTree with SMOTE
print('classification report Tuned decision tree with SMOTE')
print(classification_report(y_ts_m1, y_dt_respred))
 ##ROC score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_rf = cross_val_score(pipeline_dt, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Tuned DecisionTree with SMOTE: %.3f" % np.mean(scores_rf))
 ##savings
savings = savings_investigation_score(y_ts_m1, y_dt_respred)
print("savings Tuned DecisionTree with SMOTE: ")
print(savings)



## Voting Classifier ##
Vclf = VotingClassifier(estimators=[('decisionTree', dt_tuned), ('RandomForest', rf_tuned), ('XGBoost', Xgb_tuned)],
                        voting='soft')
Vclf.fit(X_tr_pp_m1, y_tr_m1)
y_Vcl_pred = Vclf.predict(X_ts_pp_m1)

#Evaluating voting classifier
print('classification report voting classifier')
print(classification_report(y_ts_m1, y_Vcl_pred))
 ##ROC score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_Vclf = cross_val_score(Vclf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Voting Classifier: %.3f" % np.mean(scores_Vclf))
 ##savings
savings = savings_investigation_score(y_ts_m1, y_Vcl_pred)
print("savings voting Classifier: ")
print(savings)

#Evaluating voting classifier with SMOTE
Vclf.fit(X_res, y_res)
y_Vcl_respred = Vclf.predict(X_ts_pp_m1)
pipeline_Vclf = Pipeline(steps= [('over', SMOTE()), ('VotingClassifier', Vclf)])

print('classification report voting classifier with SMOTE')
print(classification_report(y_ts_m1, y_Vcl_respred))
 ##ROC score
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_Vclf = cross_val_score(pipeline_Vclf, X, y, scoring="roc_auc_ovo", cv=cv, n_jobs=1)
print("roc_ovo Voting Classifier with SMOTE: %.3f" % np.mean(scores_Vclf))
 ##savings
savings = savings_investigation_score(y_ts_m1, y_Vcl_respred)
print("savings voting Classifier with SMOTE: ")
print(savings)