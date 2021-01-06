import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import sklearn.datasets as datesets

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics, svm

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder

from pylab import mpl

data = pd.read_csv('data_carrier_svm.csv', encoding='utf-8')
data.head()

cond = data['是否潜在合约用户']==1

data[cond]['主叫时长（分）'].hist(alpha = 0.5, label='潜在合约用户')

data[~cond]['主叫时长（分）'].hist(color = 'r', alpha = 0.5, label='非潜在合约用户')

plt.legend()

# data split

X = data.loc[:,'':'']

y = data.loc[:,'']

print('The shape of X: {0}'.format(X.shape))

print('The shape of y: {0}'.format(y.shape))

X.head()

# var service type one-hot encoding
def mapping(cell):
    if cell=='2G':
        return 2
    elif cell=='3G':
        return 3
    elif cell=='4G':
        return4

service_map = X[''].map(mapping)

service = pd.DataFrame(service_map)

enc = OneHotEncoder()

service_enc = enc.fit_transform(service).toarray()

service_names = enc.active_features_.tolist()

service_newname = [str(x)+'G' for x in service_names]

service_df = pd.DataFrame(service_enc, columns=service_newname)

service_df.head()

X_enc = pd.concat([X, service_df], axis=1).drop('',axis=1)

X_enc.head()

# normalization

from sklearn.preprocessing import normalize
X_normalized = normalize(X_enc)

# split dataset

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size = 0.2, random_state = 112)


print('The shape of X_train: {0}'.format(X_train.shape))

print('The shape of X_test: {0}'.format(X_test.shape))

plt.scatter(X_train[:,0],X_train[:,1], c=y_train)

# training

linear_clf = svm.LinearSVC()

linear_clf.fit(X_train, y_train)

y_pred = linear_clf.predict(X_test)

score = metrics.accuracy_score(y_test, y_pred)

metrics.confusion_matrix(y_test, y_pred)


