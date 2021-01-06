import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import sklearn.datasets as datesets

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDRegressor

from sklearn.metrice import r2_score

# load data
boston = datasets.load_boston()

train = boston.data

tartget = boston.target

X_train, x_test, y_train, y_true = train_test_split(train, target, test_size = 0.2)

data_df = pd.DataFrame(train, columns = boston.feature_names)

data_df

# init model

knn = KNeighborsRegressor()

linear = LinearRegression()

ridge = Ridge()

lasso = Lasso()

decision = DecisionTreeRegression()

# training

knn.fit(X_train, y_train)

linear.fit(X_train, y_train)

ridge.fit(X_train, y_train)

lasso.fit(X_train, y_train)

decision.fit(X_train, y_train)

# predict

y_pre_knn = knn.predict(x_test)

y_pre_linear = linear.predict(x_test)

y_pre_ridge = ridge.predict(x_test)

y_pre_lasso = lasso.predict(x_test)

y_pre_decision = decision.predict(x_test)

# score

knn_score = r2_score(y_true, y_pre_knn)

linear_score = r2_score(y_true, y_pre_linear)

ridge_score = r2_score(y_true, y_pre_ridge)

lasso_score = r2_score(y_true, y_pre_lasso)

decision_score = r2_score(y_true, y_pre_decision)

display(knn_score, linear_score, ridge_score, lasso_score, decision_score)

# plot 

#knn

plt.plot(y_true, label='true')
plt.plot(y_pre_knn, label='knn')
plt.legend()

plt.plot(y_true, label='true')
plt.plot(y_pre_linear, label='linear')
plt.legend()

plt.plot(y_true, label='true')
plt.plot(y_pre_ridge, label='ridge')
plt.legend()

plt.plot(y_true, label='true')
plt.plot(y_pre_lasso, label='lasso')
plt.legend()

plt.plot(y_true, label='true')
plt.plot(y_pre_decision, label='decision')
plt.legend()