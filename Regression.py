import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\Charlie\Documents\school\Thesis\INT06FI_MasterThesis.csv",index_col=1)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

y = pd.get_dummies(data["ALERT_STATUS"])
X = pd.get_dummies(data["RISK_RATING.x"])

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3, random_state= 42)

regression = LinearRegression()
regression.fit(X_train,y_train)
y_pred = regression.predict (X_test)
print(regression.score(X_test,y_test))




















