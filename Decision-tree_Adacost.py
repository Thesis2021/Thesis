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

#metacost class
# -*- coding:utf-8 -*-
from sklearn.base import clone
from adacost import AdaCost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
# Metacost Not working atm
class MetaCost(object):

    """A procedure for making error-based classifiers cost-sensitive

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> import pandas as pd
    >>> import numpy as np
    >>> S = pd.DataFrame(load_iris().data)
    >>> S['target'] = load_iris().target
    >>> LR = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    >>> C = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> model = MetaCost(S, LR, C).fit('target', 3)
    >>> model.predict_proba(load_iris().data[[2]])
    >>> model.score(S[[0, 1, 2, 3]].values, S['target'])

    .. note:: The form of the cost matrix C must be as follows:
    +---------------+----------+----------+----------+
    |  actual class |          |          |          |
    +               |          |          |          |
    |   +           | y(x)=j_1 | y(x)=j_2 | y(x)=j_3 |
    |       +       |          |          |          |
    |           +   |          |          |          |
    |predicted class|          |          |          |
    +---------------+----------+----------+----------+
    |   h(x)=j_1    |    0     |    a     |     b    |
    |   h(x)=j_2    |    c     |    0     |     d    |
    |   h(x)=j_3    |    e     |    f     |     0    |
    +---------------+----------+----------+----------+
    | C = np.array([[0, a, b],[c, 0 , d],[e, f, 0]]) |
    +------------------------------------------------+
    """
    def __init__(self, S, L, C, m=50, n=1, p=True, q=True):
        """
        :param S: The training set
        :param L: A classification learning algorithm
        :param C: A cost matrix
        :param q: Is True iff all resamples are to be used  for each examples
        :param m: The number of resamples to generate
        :param n: The number of examples in each resample
        :param p: Is True iff L produces class probabilities
        """
        if not isinstance(S, pd.DataFrame):
            raise ValueError('S must be a DataFrame object')
        new_index = list(range(len(S)))
        S.index = new_index
        self.S = S
        self.L = L
        self.C = C
        self.m = m
        self.n = len(S) * n
        self.p = p
        self.q = q

    def fit(self, flag, num_class):
        """
        :param flag: The name of classification labels
        :param num_class: The number of classes
        :return: Classifier
        """
        col = [col for col in self.S.columns if col != flag]
        S_ = {}
        M = []

        for i in range(self.m):
            # Let S_[i] be a resample of S with self.n examples
            S_[i] = self.S.sample(n=self.n, replace=True)

            X = S_[i][col].values
            y = S_[i][flag].values

            # Let M[i] = model produced by applying L to S_[i]
            model = clone(self.L)
            M.append(model.fit(X, y))

        label = []
        S_array = self.S[col].values
        for i in range(len(self.S)):
            if not self.q:
                k_th = [k for k, v in S_.items() if i not in v.index]
                M_ = list(np.array(M)[k_th])
            else:
                M_ = M

            if self.p:
                P_j = [model.predict_proba(S_array[[i]]) for model in M_]
            else:
                P_j = []
                vector = [0] * num_class
                for model in M_:
                    vector[model.predict(S_array[[i]])] = 1
                    P_j.append(vector)

            # Calculate P(j|x)
            P = np.array(np.mean(P_j, 0)).T

            # Relabel
            label.append(np.argmin(self.C.dot(P)))

        # Model produced by applying L to S with relabeled y
        X_train = self.S[col].values
        y_train = np.array(label)
        model_new = clone(self.L)
        model_new.fit(X_train, y_train)

        return model_new



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
#Construct metacost
'''
CM= np.array([[0, 0, 0], [5, 55, 55], [50, 50, 50]])

S =

CostModel = MetaCost(S, model, CM).fit('ALERT_STATUS',3)
CostModel.predict_proba(load_iris().data[[2]])
CostModel.score(S[[0, 1, 2, 3]].values, y_train)

'''
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
# Adacost
'''Check files for errors but none found 
x_train.to_excel("x_train.xlsx")
y_train.to_excel("y_train.xlsx")
'''

'''score = make_scorer(cost_calc,greater_is_better = False)'''

adac = AdaCost(algorithm = "SAMME.R", cost_matrix = CM, random_state = 100)
adac.fit(x_train,y_train)
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